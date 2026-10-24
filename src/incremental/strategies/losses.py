"""Continual learning loss functions for Naive Fine-Tuning and Learning without Forgetting (LwF)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropySeenLoss:
    """Standard Cross-Entropy loss computed over currently active/seen classes.

    Maps global target IDs to the corresponding column index in the model's
    current output head.
    """

    def __init__(self, seen_classes: List[int]) -> None:
        """Initialize CrossEntropySeenLoss.

        Args:
            seen_classes: Ordered list of global class IDs currently learned by the model.
        """
        self.seen_classes = list(seen_classes)
        # Fast lookup mapping: global_id -> head_column_index
        self._class_to_idx = {c: i for i, c in enumerate(self.seen_classes)}

    def map_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Map global class label IDs to head logit indices."""
        target_list = targets.detach().cpu().numpy().tolist()
        mapped = [self._class_to_idx[t] for t in target_list]
        return torch.tensor(mapped, dtype=torch.long, device=targets.device)

    def __call__(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss on logits and targets."""
        target_indices = self.map_targets(targets)
        return F.cross_entropy(logits, target_indices)


class LwFLoss:
    """Learning without Forgetting (LwF) composite loss.

    Formula:
        Loss = CE(all seen logits, y) + lambda_kd * KD
        where KD = T^2 * KL( Softmax(teacher_old_logits / T) || Softmax(student_old_logits / T) )

    The teacher is frozen in eval mode and evaluated on the same augmented new-task
    batch as the student.
    """

    def __init__(
        self,
        teacher: Optional[nn.Module],
        seen_classes: List[int],
        num_old_classes: int,
        temperature: float = 2.0,
        lambda_kd: float = 1.0,
    ) -> None:
        """Initialize LwFLoss.

        Args:
            teacher: Frozen teacher model from previous task (or None for Task 1).
            seen_classes: Ordered list of global class IDs seen so far.
            num_old_classes: Number of classes in previous task.
            temperature: Distillation temperature T (default 2.0).
            lambda_kd: Distillation loss weighting factor lambda (default 1.0).
        """
        self.teacher = teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.seen_classes = list(seen_classes)
        self.num_old_classes = num_old_classes
        self.temperature = temperature
        self.lambda_kd = lambda_kd
        self._class_to_idx = {c: i for i, c in enumerate(self.seen_classes)}

    def map_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Map global class label IDs to head logit indices."""
        target_list = targets.detach().cpu().numpy().tolist()
        mapped = [self._class_to_idx[t] for t in target_list]
        return torch.tensor(mapped, dtype=torch.long, device=targets.device)

    def __call__(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute composite CE + KD distillation loss."""
        # 1. Classification loss on all seen classes
        target_indices = self.map_targets(targets)
        ce_loss = F.cross_entropy(logits, target_indices)

        # 2. Knowledge distillation loss (if teacher present and old classes exist)
        if self.teacher is not None and self.num_old_classes > 0:
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                if teacher_logits.shape[1] > self.num_old_classes:
                    teacher_logits = teacher_logits[:, : self.num_old_classes]

            # Student old-class logits (first num_old_classes outputs)
            student_old_logits = logits[:, : self.num_old_classes]

            # In PyTorch F.kl_div(input, target): input is log-probs, target is probs
            log_p_student = F.log_softmax(student_old_logits / self.temperature, dim=1)
            p_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

            kd_loss = (self.temperature ** 2) * F.kl_div(
                log_p_student,
                p_teacher,
                reduction="batchmean",
            )
            return ce_loss + (self.lambda_kd * kd_loss)

        return ce_loss


class LwFNewLoss:
    """LwF variant computing Cross-Entropy over NEW-class logits only + KD on old logits.

    Prevents cross-entropy gradient from directly pushing down old-class logits.
    """

    def __init__(
        self,
        teacher: Optional[nn.Module],
        seen_classes: List[int],
        num_old_classes: int,
        temperature: float = 2.0,
        lambda_kd: float = 1.0,
    ) -> None:
        self.teacher = teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.seen_classes = list(seen_classes)
        self.num_old_classes = num_old_classes
        self.temperature = temperature
        self.lambda_kd = lambda_kd
        self._class_to_idx = {c: i for i, c in enumerate(self.seen_classes)}

    def map_targets(self, targets: torch.Tensor) -> torch.Tensor:
        target_list = targets.detach().cpu().numpy().tolist()
        mapped = [self._class_to_idx[t] for t in target_list]
        return torch.tensor(mapped, dtype=torch.long, device=targets.device)

    def __call__(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        target_indices = self.map_targets(targets)

        if self.num_old_classes > 0:
            # Cross-entropy over new classes only
            new_logits = logits[:, self.num_old_classes:]
            new_targets = target_indices - self.num_old_classes
            ce_loss = F.cross_entropy(new_logits, new_targets)
        else:
            ce_loss = F.cross_entropy(logits, target_indices)

        if self.teacher is not None and self.num_old_classes > 0:
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                if teacher_logits.shape[1] > self.num_old_classes:
                    teacher_logits = teacher_logits[:, : self.num_old_classes]

            student_old_logits = logits[:, : self.num_old_classes]
            log_p_student = F.log_softmax(student_old_logits / self.temperature, dim=1)
            p_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

            kd_loss = (self.temperature ** 2) * F.kl_div(
                log_p_student,
                p_teacher,
                reduction="batchmean",
            )
            return ce_loss + (self.lambda_kd * kd_loss)

        return ce_loss


class LwFLLoss:
    """Variant V2 'lwfL': CE over all seen logits + lambda * MSE on RAW old logits.

    Penalizes absolute shift of old logits between student and frozen teacher.
    """

    def __init__(
        self,
        teacher: Optional[nn.Module],
        seen_classes: List[int],
        num_old_classes: int,
        lambda_kd: float = 1.0,
    ) -> None:
        self.teacher = teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.seen_classes = list(seen_classes)
        self.num_old_classes = num_old_classes
        self.lambda_kd = lambda_kd
        self._class_to_idx = {c: i for i, c in enumerate(self.seen_classes)}

    def map_targets(self, targets: torch.Tensor) -> torch.Tensor:
        target_list = targets.detach().cpu().numpy().tolist()
        mapped = [self._class_to_idx[t] for t in target_list]
        return torch.tensor(mapped, dtype=torch.long, device=targets.device)

    def __call__(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        target_indices = self.map_targets(targets)
        ce_loss = F.cross_entropy(logits, target_indices)

        if self.teacher is not None and self.num_old_classes > 0:
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                if teacher_logits.shape[1] > self.num_old_classes:
                    teacher_logits = teacher_logits[:, : self.num_old_classes]

            student_old_logits = logits[:, : self.num_old_classes]
            mse_loss = F.mse_loss(student_old_logits, teacher_logits)
            return ce_loss + (self.lambda_kd * mse_loss)

        return ce_loss


class LwFLFLoss:
    """Variant V3 / V4 'lwfLF': V2 + mu * (1 - cosine_similarity) on pooled features.

    Penalizes both raw old-logit drift and pooled feature representation drift.
    """

    def __init__(
        self,
        teacher: Optional[nn.Module],
        seen_classes: List[int],
        num_old_classes: int,
        lambda_kd: float = 1.0,
        mu: float = 1.0,
    ) -> None:
        self.teacher = teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.seen_classes = list(seen_classes)
        self.num_old_classes = num_old_classes
        self.lambda_kd = lambda_kd
        self.mu = mu
        self._class_to_idx = {c: i for i, c in enumerate(self.seen_classes)}

    def map_targets(self, targets: torch.Tensor) -> torch.Tensor:
        target_list = targets.detach().cpu().numpy().tolist()
        mapped = [self._class_to_idx[t] for t in target_list]
        return torch.tensor(mapped, dtype=torch.long, device=targets.device)

    def __call__(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        target_indices = self.map_targets(targets)
        ce_loss = F.cross_entropy(logits, target_indices)

        if self.teacher is not None and self.num_old_classes > 0:
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                if teacher_logits.shape[1] > self.num_old_classes:
                    teacher_logits = teacher_logits[:, : self.num_old_classes]
                teacher_features = self.teacher.extract_features(images)

            student_old_logits = logits[:, : self.num_old_classes]
            mse_loss = F.mse_loss(student_old_logits, teacher_logits)

            student_features = model.extract_features(images)
            cos_sim = F.cosine_similarity(student_features, teacher_features, dim=-1)
            feat_loss = (1.0 - cos_sim).mean()

            return ce_loss + (self.lambda_kd * mse_loss) + (self.mu * feat_loss)

        return ce_loss
