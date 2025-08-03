from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
import logging
import transformers
import json
import os
from typing import List, Dict, Any
from peft import PeftModel


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(data: List[Dict[str, Any]], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_llm(model_id, base_model_name, device, quantize=False, batch_size=1, is_lora=False, lora_weights=None, return_full_text=True, max_new_tokens=1024, temperature = 0.8):
    # Silence warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("tokenizers").setLevel(logging.ERROR)
    transformers.logging.set_verbosity_error()

    config = AutoConfig.from_pretrained(base_model_name)
    config.use_cache = False
    config.gradient_checkpointing = True

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded.")

    # 2. Prepare quantization config if needed
    quant_config = None
    if quantize:
        print("⚙️  Using 8-bit quantization.")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False  # reduces memory peak
        )

    # 3. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        device_map=device,
        trust_remote_code=True,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
    )
    print("✅ Base model loaded.")

    # 4. Apply LoRA if needed
    if is_lora and lora_weights:
        print(f"🔗 Loading LoRA adapter from {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights)
        model = model.merge_and_unload()
        print("✅ LoRA merged into base model.")

    # 5. Resize token embeddings with memory-efficient option
    try:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
    except:
        print("⚠️ Token resizing failed; continuing without it.")
    model.config.pad_token_id = tokenizer.pad_token_id

    # 6. Return pipeline (or switch to manual generation if needed)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        top_k=30,
        temperature=temperature,
        batch_size=batch_size,
        return_full_text=return_full_text,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    return pipe, tokenizer
