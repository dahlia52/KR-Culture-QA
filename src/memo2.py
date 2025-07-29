# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "LGAI-EXAONE/EXAONE-4.0-32B"

# device = "cuda:1"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="bfloat16",
#     device_map=device
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# messages = [
#     {"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}
# ]
# input_ids = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     enable_thinking=True,
# )

# output = model.generate(
#     input_ids.to(device),
#     max_new_tokens=128,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.95
# )
# print(tokenizer.decode(output[0]))




# import os
# import gc
# import argparse
# import json
# import tqdm
# from typing import List, TypedDict, Optional
# import torch
# from datasets import load_from_disk
# from langchain.vectorstores import Chroma
# from langchain_core.documents import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     pipeline,
#     BitsAndBytesConfig
# )
# from make_prompt import *
# from retrieve import *
# from load import *
# from generators import *
# import logging
# from datetime import datetime
# from peft import PeftModel, PeftConfig

# RETRIEVER_NAME = "jinaai/jina-embeddings-v3"
# GENERATOR_NAME = "K-intelligence/Midm-2.0-Base-Instruct"
# base_model_name = "K-intelligence/Midm-2.0-Base-Instruct"
# CHROMA_DB_PATH = "resource/retrieval_docs/chroma_db_jina"
# KOWIKI_DATASET_PATH = "resource/retrieval_docs/kowiki_dataset"

# print(f"Current device: {torch.cuda.get_device_name(0)}")
# print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# device = "cuda:0"
# torch.cuda.reset_peak_memory_stats(device)

# print("=" * 50)
# print("Starting Korean Culture QA System")
# print("=" * 50)
# retriever = load_retriever(model=RETRIEVER_NAME, device=device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=10)

# question = "문화지식, 가치관, 한국에서는 밥을 먹고 바로 누우면 무엇이 된다고 생각하나요?"
# documents = retriever.invoke(question)
# logging.info(f"Number of retrieved documents: {len(documents)}")
# context = format_docs(documents)
# print(context)


q = "강원도 화천에서 추운 겨울철에 언 강에서 열리는 축제의 이름은 무엇인가요? \\n 1\\t 산천어 축제  2\\t 얼음 축제  3\\t 송사리 축제  4\\t 빙어 축제  5\\t 광한 축제"
a = "<reasoning>한국의 전통적인 관습에 따르면 장례식과 같은 부정(不淨)한 장소를 방문한 후에는 집 안으로 들어가기 전에 부정을 씻어내기 위한 행위를 한다. 주어진 문서에서 언급된 바와 같이, 장례식에 다녀온 후 집에 들어갈 때 소금을 뿌리는 것이 귀신이나 부정한 기운을 막기 위한 방법으로 알려져 있다. 이는 소금이 정화력을 가지고 있다고 믿었기 때문이다. 설탕이나 간장, 된장은 이러한 목적으로 사용되지 않는다.\u0000assistant3\u0000\u0000\u0000"
print(a.split("assistant")[1][0])
