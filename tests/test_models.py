"""Unit tests for Custom CNN model architectures, expandable heads, and profiling."""

import pytest
import torch

from src.models.custom_cnn.factory import VARIANT_CONFIGS, build_model
from src.models.custom_cnn.model import CustomCNN
from src.models.heads.expandable_head import ExpandableLinearHead
from src.models.utils.profiling import (
    benchmark_latency,
    count_flops,
    count_parameters,
    get_model_size_mb,
)


@pytest.mark.parametrize("variant", ["small", "medium", "large"])
def test_every_variant_builds_under_5m_params(variant: str):
    """Verify each variant instantiates with 4 classes and stays strictly under 5M parameters."""
    model = build_model(variant=variant, num_classes=4)
    assert isinstance(model, CustomCNN)

    param_info = count_parameters(model)
    total_params = param_info["total"]
    assert total_params < 5_000_000, f"Variant {variant} breached 5M budget: {total_params}"
    assert param_info["trainable"] == total_params
    assert param_info["head"] > 0
    assert param_info["backbone"] > 0


@pytest.mark.parametrize("variant", ["small", "medium", "large"])
def test_forward_pass_output_shape(variant: str):
    """Verify forward pass on (2, 3, 150, 150) tensor returns expected logits shape (2, num_classes)."""
    model = build_model(variant=variant, num_classes=3)
    model.eval()

    dummy_input = torch.randn(2, 3, 150, 150)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in model outputs"


def test_expand_head_bit_identical_weights():
    """Verify expand() maintains exact bit-identical weights and biases for previous classes."""
    head = ExpandableLinearHead(in_features=64, initial_num_classes=2)

    old_weight = head.linear.weight.clone()
    old_bias = head.linear.bias.clone()

    head.expand(n_new_classes=1)

    assert head.num_classes == 3
    assert head.linear.weight.shape == (3, 64)
    assert head.linear.bias.shape == (3,)

    # Test bit-identical copy of previous class weights and biases
    assert torch.equal(head.linear.weight[:2], old_weight), "Previous weights were modified during expand()"
    assert torch.equal(head.linear.bias[:2], old_bias), "Previous biases were modified during expand()"


def test_expand_keeps_old_logits_unchanged_in_eval_mode():
    """Verify expand() preserves old-class output logits for identical inputs in eval mode."""
    torch.manual_seed(42)
    model = build_model(variant="small", num_classes=2)
    model.eval()

    dummy_input = torch.randn(2, 3, 150, 150)

    with torch.no_grad():
        initial_logits = model(dummy_input)

    assert initial_logits.shape == (2, 2)

    # Expand by 1 class (2 -> 3 classes)
    model.expand(n_new_classes=1)
    assert model.num_classes == 3

    with torch.no_grad():
        expanded_logits = model(dummy_input)

    assert expanded_logits.shape == (2, 3)

    # Old logits should be bit-identical or numerically indistinguishable
    old_logits_extracted = expanded_logits[:, :2]
    assert torch.allclose(initial_logits, old_logits_extracted, atol=1e-7), (
        "Old class logits shifted after head expansion in eval mode"
    )

    # Also test model.forward_old_logits helper
    with torch.no_grad():
        old_logits_helper = model.forward_old_logits(dummy_input, num_old_classes=2)
    assert torch.equal(old_logits_extracted, old_logits_helper)


def test_expand_sequence_across_tasks():
    """Verify head correctly expands through the class incremental sequence: 2 -> 3 -> 4."""
    model = build_model(variant="small", num_classes=2)
    assert model.num_classes == 2

    # Task 2 arrival: +1 class -> 3 classes
    model.expand(n_new_classes=1)
    assert model.num_classes == 3

    # Task 3 arrival: +1 class -> 4 classes
    model.expand(n_new_classes=1)
    assert model.num_classes == 4

    dummy = torch.randn(1, 3, 150, 150)
    out = model(dummy)
    assert out.shape == (1, 4)


def test_feature_extraction():
    """Verify extract_features separates representation learning from the classification head."""
    model = build_model(variant="small", num_classes=4)
    model.eval()

    dummy = torch.randn(2, 3, 150, 150)
    with torch.no_grad():
        features = model.extract_features(dummy)
        logits = model(dummy)
        head_logits = model.head(features)

    assert features.shape == (2, model.feature_dim)
    # Head applied to features matches end-to-end forward pass in eval mode (with dropout p=0 effect)
    assert torch.allclose(logits, head_logits, atol=1e-7)


def test_forward_backward_on_gpu():
    """Verify forward and backward gradients execute on CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this environment")

    device = torch.device("cuda")
    model = build_model(variant="small", num_classes=4).to(device)
    model.train()

    dummy_input = torch.randn(2, 3, 150, 150, device=device, requires_grad=True)
    target = torch.tensor([0, 2], device=device, dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss()
    logits = model(dummy_input)
    loss = criterion(logits, target)
    loss.backward()

    # Verify gradients computed for all conv layers and head
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has None grad"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
