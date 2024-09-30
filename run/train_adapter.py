import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig

from src.data import CustomDataset, DataCollatorForSupervisedDataset

torch.cuda.empty_cache()

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, help="scheduler warmup steps")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=5, help="training epoch")
g.add_argument("--use_adapter", action='store_true', help="whether to use adapters during training")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_adapter:
        # Define the adapter configuration (using LoRA here, but you can customize)
        adapter_config = LoraConfig(
            r=16,  # Rank of the adapters
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.1,  # Dropout probability
            bias="none",  # Bias setting
            target_modules=["q_proj", "v_proj"],  # Modules to apply adapters to
        )
        # Apply the adapter configuration to the model

    train_dataset = CustomDataset("resource/data/일상대화요약_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/일상대화요약_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.1,
        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        log_level="info",
        logging_steps=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024,
        packing=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        peft_config=adapter_config,
        compute_metrics = compute_metrics #
    )

    trainer.train()

    if args.use_adapter:
        # Save the adapter after training
        model.save_pretrained(args.save_dir + "/adapter__1")
#
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    loss_fct = torch.nn.CrossEntropyLoss()
    eval_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    accuracy = (predictions == labels).float().mean()
    
    return {
        "eval_loss": eval_loss.item(),
        "accuracy": accuracy.item(),
    }

if __name__ == "__main__":
    exit(main(parser.parse_args()))
