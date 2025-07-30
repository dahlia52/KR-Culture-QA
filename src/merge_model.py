from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

#GENERATOR = "./models/fine-tuned-model-rationale-선다형_to_서술형-sorted-without-MC-NEW"
GENERATOR = "./models/fine-tuned-model-선다형-단답형-서술형-NEW"

is_lora = os.path.isdir(GENERATOR) and 'adapter_config.json' in os.listdir(GENERATOR)
lora_weights = GENERATOR if is_lora else None
    
if is_lora:
    print(f"🔍 Detected a fine-tuned LoRA model at: {GENERATOR}")
    config = PeftConfig.from_pretrained(GENERATOR)
    base_model_name = config.base_model_name_or_path
    print(f"🔧 Loading base model: {base_model_name}")

tokenizer = AutoTokenizer.from_pretrained(GENERATOR, trust_remote_code=True)
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="cuda:1",
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

if is_lora and lora_weights:
    print(f"Loading LoRA weights from {lora_weights}")
    model = PeftModel.from_pretrained(model, lora_weights)
    model = model.merge_and_unload()
    model = model.half()

model.save_pretrained("./models/fine-tuned-model-선다형-단답형-서술형-NEW-merged-float16")