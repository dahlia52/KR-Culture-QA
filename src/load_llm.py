from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

def load_llm(model_id, device, quantize=False, batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if quantize:
        print("Quantizing model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        top_k=30,
        temperature=0.8,
        batch_size=batch_size
    )

    return pipe
