# Unit and Integration Tests (`tests/`)

This directory houses automated test suites designed to verify pipeline integrity, data loading, model architecture constraints, incremental learning mechanics, and evaluation calculations.

## Test Organization
Test files mirror the structure of `src/`:
- `test_seed.py`: Verifies determinism across Python, NumPy, and PyTorch (CPU and CUDA).
- `test_config.py`: Verifies configuration loading, schema validation, and parameter consistency.
- `test_data.py`: Verifies dataset loaders, extreme contour crop integrity, and stratified split non-overlap.
- `test_models.py`: Verifies parameter limits (<5M params for Custom CNN), tensor output shapes, and expandable heads.
- `test_incremental.py`: Verifies LwF distillation loss, teacher freezing, and buffer management in Experience Replay.
- `test_metrics.py`: Verifies mathematical correctness of Average Incremental Accuracy, Forgetting, and Backward Transfer.

## Running Tests
Execute test suites via `pytest`:
```bash
pytest tests/ -v
```
