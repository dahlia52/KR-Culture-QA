import os
import argparse
import json
import tqdm
from typing import List, TypedDict, Optional
import torch
from datasets import load_from_disk
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from make_prompt import *
from retrieve import *

# Generate Answers
def generate(args, retriever, pipe, result_data):
    prompts = []
    system_prompt = make_system_prompt()
    
    print("Preparing prompts...")
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        context = ""
        if args.retrieve or args.retrieve_adaptively:
            print("Retrieving relevant documents...")
            documents = retriever.invoke(question)
            print("Number of retrieved documents:", len(documents))
            context = format_docs(documents)
        
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=item["input"]["topic_keyword"],
            context=context,
            question=question,
            fewshot=True,
            retrieve = args.retrieve or args.retrieve_adaptively
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # pipeline's tokenizer will apply the chat template
        prompts.append(messages)

    print("Generating answers in batch...")
    outputs = pipe(prompts)

    print("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        # The output from the pipeline is a list with a dictionary
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        result_data[idx]["output"] = {"answer": answer.strip()}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False, indent=4))


# Self-Reflection
def generate_for_self_reflection(args, retriever, pipe, result_data):
    prompts = []
    system_prompt = make_system_prompt()

    print("Preparing prompts...")
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        context = ""
        if args.retrieve or args.retrieve_adaptively:
            print("Retrieving relevant documents...")
            documents = retriever.invoke(question)
            context = format_docs(documents)
        
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=item["input"]["topic_keyword"],
            context=context,
            question=question,
            fewshot=True,
            retrieve=args.retrieve or args.retrieve_adaptively,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # pipeline's tokenizer will apply the chat template
        prompts.append(messages)

    print("Generating answers in batch...")
    outputs = pipe(prompts)

    print("Processing generated answers...")
    prompts = []
    for output in tqdm.tqdm(outputs):
        generated_text = output[0]['generated_text']
        instruction = output[0]['generated_text'][1]['content'].split("[지침]\n")[1].split("\n\n")[0]
        question = output[0]['generated_text'][1]['content'].split("[질문]\n")[1].split("[답변]")[0]
        answer = generated_text[-1]['content']
        user_prompt = make_prompt_for_reflection(
            question=question,
            instruction=instruction,
            answer=answer,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompts.append(messages)
        
    outputs = pipe(prompts)
    
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        try:
            answer = output[0]['generated_text'][-1]['content'].split("<answer>")[1].split("</answer>")[0].replace('\n', '').strip()
            
        except:
            try:
                answer = output[0]['generated_text'][-1]['content'].split("<answer>")[1].replace('\n', '').strip()
            except:
                try:
                    answer = output[0]['generated_text'][-1]['content'].split('/reasoning')[1].replace('\n', '').strip()
                except:
                    answer = output[0]['generated_text'][-1]['content'].replace('\n', '').strip()
        result_data[idx]['output'] = {"answer": answer.strip()}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False, indent=4))




# Consider logits for multiple choice
def generate_for_multiple_choice(args, retriever, pipe, result_data):
    prompts = []
    other_indices = []
    system_prompt = make_system_prompt()

    print("Preparing prompts...")
    for idx, item in enumerate(tqdm.tqdm(result_data)):
        question = item["input"]["question"]
        question_type = item["input"]["question_type"]
        context = ""

        if args.retrieve or args.retrieve_adaptively:
            documents = retriever.invoke(question)
            context = format_docs(documents)

        if question_type == '선다형':
            user_prompt_base = make_prompt(
                question_type=question_type,
                category=item["input"]["category"],
                domain=item["input"]["domain"],
                topic_keyword=item["input"]["topic_keyword"],
                context=context,
                question=question,
                fewshot=False,
                retrieve=args.retrieve or args.retrieve_adaptively
            )

            messages_base = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_base}
            ]
            
            prompt_text = pipe.tokenizer.apply_chat_template(messages_base, tokenize=False, add_generation_prompt=True)

            choices = ["1"]
            if "2\\t" in question:
                choices.append("2")
            if "3\\t" in question:
                choices.append("3")
            if "4\\t" in question:
                choices.append("4")
            if "5\\t" in question:
                choices.append("5")

            choice_scores = []
            for choice in choices:
                with torch.no_grad():
                    inputs = pipe.tokenizer(prompt_text).input_ids
                    choice_tokens = pipe.tokenizer(choice, add_special_tokens=False).input_ids
                    full_input = pipe.tokenizer(prompt_text + choice, return_tensors="pt").to(args.device)

                    labels = torch.tensor([[-100] * len(inputs) + choice_tokens]).to(args.device)
                    outputs = pipe.model(**full_input, labels=labels)
                    logits = outputs.logits
                    
                    # Calculate score for the choice
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # The logits are shifted by one position
                    relevant_log_probs = log_probs[0, -len(choice_tokens)-1:-1, :]
                    choice_token_ids = torch.tensor(choice_tokens).to(args.device)
                    
                    score = relevant_log_probs.gather(1, choice_token_ids.unsqueeze(-1)).squeeze(-1).sum().item()
                    choice_scores.append(score)

            best_choice_idx = torch.argmax(torch.tensor(choice_scores)).item()
            answer = str(best_choice_idx + 1)
            result_data[idx]["output"] = {"answer": answer}

        else:
            if args.retrieve or args.retrieve_adaptively:
                documents = retriever.invoke(question)
                context = format_docs(documents)
            # For other question types, use the original batch processing
            user_prompt = make_prompt(
                question_type=question_type,
                category=item["input"]["category"],
                domain=item["input"]["domain"],
                topic_keyword=item["input"]["topic_keyword"],
                context=context,
                question=question,
                fewshot=True,
                retrieve=args.retrieve or args.retrieve_adaptively
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompts.append(messages)
            other_indices.append(idx)

    if prompts:
        print("Generating answers for non-multiple-choice questions in batch...")
        outputs = pipe(prompts)

        print("Processing generated answers...")
        for i, output in enumerate(tqdm.tqdm(outputs)):
            original_idx = other_indices[i]
            generated_text = output[0]['generated_text']
            # The actual answer is the last message in the generated text
            if isinstance(generated_text, list) and generated_text:
                answer = generated_text[-1]['content']
            else: # Fallback for different output formats
                answer = str(generated_text)
            result_data[original_idx]["output"] = {"answer": answer.strip()}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False, indent=4))