"""
ATHENA Training Module
======================
Centralized training for all ATHENA components.

Training Pipeline:
- Stage 1: Fine-tune OLMoE 1B on finance data
- Stage 2: Train AgeMem with Step-wise GRPO
"""

from .stage1_finetune.finetune import FinanceFineTuner
from .stage2_agemem.trainer import AgeMemTrainer

__all__ = [
    "FinanceFineTuner",
    "AgeMemTrainer",
]
