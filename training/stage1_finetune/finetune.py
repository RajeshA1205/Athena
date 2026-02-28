"""
Finance Fine-tuning Script
==========================
Fine-tune OLMoE 1B on comprehensive finance data.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import logging

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
    )
    from datasets import Dataset, load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from .config import FineTuneConfig


class FinanceFineTuner:
    """
    Finance Fine-tuner for OLMoE.

    Stage 1 of ATHENA training pipeline:
    - Loads OLMoE 1B base model
    - Applies LoRA for parameter-efficient training
    - Fine-tunes on scraped finance data (news, market, social)
    """

    def __init__(self, config: Optional[FineTuneConfig] = None):
        """
        Initialize fine-tuner.

        Args:
            config: Fine-tuning configuration
        """
        self.config = config or FineTuneConfig()
        self.logger = logging.getLogger("athena.training.finetune")

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self) -> bool:
        """
        Set up model and tokenizer.

        Returns:
            True if setup successful
        """
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("transformers not installed")
            return False

        if self.config.use_lora and not PEFT_AVAILABLE:
            self.logger.error("peft not installed but use_lora=True")
            return False

        try:
            self.logger.info("Loading model: %s", self.config.model_name)

            # Quantization config
            bnb_config = None
            if self.config.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Prepare for k-bit training
            if self.config.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            # Apply LoRA
            if self.config.use_lora:
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()

            self.logger.info("Model setup complete")
            return True

        except Exception as e:
            self.logger.error("Setup failed: %s", e)
            return False

    def prepare_dataset(
        self,
        train_data: Optional[Dataset] = None,
        eval_data: Optional[Dataset] = None,
    ) -> Dict[str, Dataset]:
        """
        Prepare training and evaluation datasets.

        Args:
            train_data: Training dataset (or load from config path)
            eval_data: Evaluation dataset (or load from config path)

        Returns:
            Dictionary with "train" and "eval" datasets
        """
        datasets = {}

        # Load from paths if not provided
        if train_data is None and self.config.train_data_path:
            train_data = load_dataset("json", data_files=self.config.train_data_path)["train"]

        if eval_data is None and self.config.eval_data_path:
            eval_data = load_dataset("json", data_files=self.config.eval_data_path)["train"]

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )

        if train_data is not None:
            datasets["train"] = train_data.map(
                tokenize,
                batched=True,
                remove_columns=train_data.column_names,
            )

        if eval_data is not None:
            datasets["eval"] = eval_data.map(
                tokenize,
                batched=True,
                remove_columns=eval_data.column_names,
            )

        return datasets

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """
        Run fine-tuning.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Training metrics
        """
        if self.model is None:
            raise RuntimeError("Call setup() first")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=self.config.report_to,
            seed=self.config.seed,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        self.logger.info("Starting training...")
        train_result = self.trainer.train()

        # Save
        self.save()

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }

    def save(self, path: Optional[str] = None) -> None:
        """
        Save fine-tuned model.

        Args:
            path: Save path (uses config.output_dir if not provided)
        """
        save_path = path or self.config.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if self.config.use_lora:
            # Save only LoRA weights
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(save_path)

        self.tokenizer.save_pretrained(save_path)
        self.logger.info("Model saved to %s", save_path)

    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Call train() first")

        return self.trainer.evaluate(eval_dataset)
