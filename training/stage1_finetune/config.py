"""
Fine-tuning Configuration
=========================
Configuration for Stage 1 finance fine-tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FineTuneConfig:
    """Configuration for OLMoE finance fine-tuning."""

    # Model
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    output_dir: str = "./checkpoints/stage1_finetune"

    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Sequence length
    max_seq_length: int = 2048
    packing: bool = False  # Pack multiple samples into one sequence

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100

    # Data
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    data_mix_ratios: dict = field(default_factory=lambda: {
        "news": 0.4,      # News & SEC filings
        "market": 0.3,    # Market data analysis
        "social": 0.3,    # Social/sentiment
    })

    # Misc
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    report_to: str = "none"  # "wandb", "tensorboard", "none"
