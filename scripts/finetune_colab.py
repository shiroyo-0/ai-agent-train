#!/usr/bin/env python3
"""
=== SHIRO Nb.1.0 FINE-TUNING ===
Jalankan di Google Colab atau Kaggle (butuh GPU).

Steps:
1. Upload file ini ke Colab/Kaggle
2. Run semua cell
3. Download model hasil fine-tune
4. Deploy ke VPS

Colab: Runtime → Change runtime type → GPU (T4)
Kaggle: Settings → Accelerator → GPU T4 x2
"""

# === CELL 1: Install dependencies ===
# !pip install -q torch transformers peft datasets accelerate bitsandbytes trl huggingface_hub

# === CELL 2: Setup ===
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./shiro-nb1-finetuned"

# === CELL 3: Download training data from GitHub ===
# !git clone https://github.com/shiroyo-0/ai-agent-train.git
# !find ai-agent-train/data/training -name "*.jsonl" -exec cat {} \; > all_training_data.jsonl

# Or upload manually - combine all .jsonl files into one

# === CELL 4: Load and prepare dataset ===
def load_training_data(path="all_training_data.jsonl"):
    examples = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get("instruction") and d.get("output") and len(d["output"]) > 20:
                    examples.append({
                        "text": f"<|im_start|>system\nYou are Shiro Nb.1.0, a helpful AI assistant built by Shiro. You are friendly, creative, and knowledgeable.<|im_end|>\n<|im_start|>user\n{d['instruction']}<|im_end|>\n<|im_start|>assistant\n{d['output']}<|im_end|>"
                    })
            except:
                continue
    return Dataset.from_list(examples)

dataset = load_training_data()
print(f"Training examples: {len(dataset)}")

# === CELL 5: Load model with 4-bit quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# === CELL 6: LoRA config ===
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === CELL 7: Training ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=1024,
)

print("🎓 Starting fine-tuning...")
trainer.train()

# === CELL 8: Save model ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")

# === CELL 9: Merge LoRA weights into base model (for deployment) ===
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True)
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()

MERGED_DIR = "./shiro-nb1-merged"
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"✅ Merged model saved to {MERGED_DIR}")

# === CELL 10: Upload to HuggingFace (optional) ===
# from huggingface_hub import login, HfApi
# login(token="YOUR_HF_TOKEN")
# merged_model.push_to_hub("shiroyo-0/shiro-nb1")
# tokenizer.push_to_hub("shiroyo-0/shiro-nb1")

# === CELL 11: Test the model ===
print("\n🧪 Testing Shiro Nb.1.0...")
inputs = tokenizer("<|im_start|>system\nYou are Shiro Nb.1.0<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(merged_model.device)
with torch.no_grad():
    out = merged_model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# === DONE ===
# Download folder "shiro-nb1-merged" dan upload ke VPS:
# scp -r shiro-nb1-merged/ root@165.22.102.81:/root/ai-agent/models/shiro-nb1/
#
# Lalu di VPS, update serve_and_train.py:
# MODEL_NAME = "/root/ai-agent/models/shiro-nb1"
# systemctl restart shiro-ai
