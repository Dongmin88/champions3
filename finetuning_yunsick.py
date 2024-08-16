import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
import huggingface_hub

# Hugging Face login (make sure to keep your token secure)
huggingface_hub.login("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")

# Model and dataset paths
base_model = "beomi/Llama-3-Open-Ko-8B"
new_model = "Llama3-Ko-3-8B-baemin"

# Load the dataset from CSV
dataset = load_dataset('csv', data_files='train.csv', split='train')

# Data preprocessing
def preprocess_qa(example):
    return {
        'text': f"Question: {example['Question']} Answer: {example['Answer']}"
    }

processed_dataset = dataset.map(preprocess_qa, remove_columns=['Question', 'Answer'])

# Check CUDA capabilities
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# QLoRA config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=5,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Set up trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Start training
trainer.train()

# Save the model
model_save_path = "./results/final_model"
trainer.save_model(model_save_path)

print("Training completed and model saved.")

# Inference code (you may want to run this in a separate script)
def generate_answer(question, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=5,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split('?')[1].strip() if '?' in answer else answer.strip()

# Example of using the generate_answer function
# from pandas import read_csv

# test_df = read_csv('test.csv')
# submission_df = read_csv('sample_submission.csv')
# submission_df['Answer'] = test_df['Question'].apply(lambda q: generate_answer(q, "./results/final_model"))
# submission_df.to_csv("submission_results.csv", index=False)