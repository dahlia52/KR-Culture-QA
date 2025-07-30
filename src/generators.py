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
import logging
from load import save_dataset


# Generate Answers
def generate(args, retriever, pipe, result_data, contexts):
    logging.info("### Generate answers ###")
    prompts = []
    system_prompt = make_system_prompt()
    
    logging.info("Preparing prompts...")
    
    for idx, item in tqdm.tqdm(enumerate(result_data)):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
            context=contexts[idx],
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

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        answer = answer.replace("assistant", "")
        answer = answer.replace("\u0000", "")
        if '답변:' in answer:
            answer = answer.split('답변:')[1].strip()
        result_data[idx]["input"]["context"] = contexts[idx].strip()
        result_data[idx]["output"] = {"answer": answer.strip()}

    return result_data


# Generate Answers
def generate_with_rationale(args, retriever, pipe, result_data):
    logging.info("### Generate answers ###")
    prompts = []
    contexts = []
    system_prompt = make_system_prompt()
    
    logging.info("Preparing prompts...")
    
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]

        context = ""
        if args.retrieve or args.retrieve_adaptively:
            context = retrieve_documents(topic_keyword, question, retriever)
        
        contexts.append(context)
        
        user_prompt = make_prompt_for_rationale(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
            context=context,
            question=question,
            fewshot=True,
            retrieve = args.retrieve or args.retrieve_adaptively
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompts.append(messages)

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        answer = answer.replace("assistant", "")
        answer = answer.replace("\u0000", "")
        if '답변:' in answer:
            answer = answer.split('답변:')[1].strip()
        result_data[idx]["input"]["context"] = contexts[idx].strip()
        try:
            answer = generated_text[-1]['content'].split("<answer>")[1].split("</answer>")[0].replace('\n', '').strip()
            
        except:
            try:
                answer = generated_text[-1]['content'].split("<answer>")[1].replace('\n', '').strip()
            except:
                try:
                    answer = generated_text[-1]['content'].split('/reasoning')[1].replace('\n', '').strip()
                except:
                    answer = generated_text[-1]['content'].replace('\n', '').strip()
        result_data[idx]["output"] = {"answer": answer.strip()}

    return result_data






# Self-Reflection
def generate_for_self_reflection(args, retriever, pipe, result_data):
    logging.info("### Generate for self reflection ###")
    prompts = []
    system_prompt = make_system_prompt()
    topic_keywords = []

    logging.info("Preparing prompts...")
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        topic_keywords.append(topic_keyword)
        context = ""
        if args.retrieve or args.retrieve_adaptively:
            context = retrieve_documents(topic_keyword, question, retriever)
        
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
            context=context,
            question=question,
            fewshot=True,
            retrieve=args.retrieve or args.retrieve_adaptively,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompts.append(messages)

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    prompts = []
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        instruction = generated_text[1]['content'].split("[지침]\n")[1].split("\n\n")[0]
        question = generated_text[1]['content'].split("[질문]\n")[1].split("[답변]")[0]
        answer = generated_text[-1]['content']
        user_prompt = make_prompt_for_reflection(
            question=question,
            instruction=instruction,
            answer=answer,
            topic_keyword=topic_keywords[idx],
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompts.append(messages)
        
    outputs = pipe(prompts)
    
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        try:
            answer = generated_text[-1]['content'].split("<answer>")[1].split("</answer>")[0].replace('\n', '').strip()
            
        except:
            try:
                answer = generated_text[-1]['content'].split("<answer>")[1].replace('\n', '').strip()
            except:
                try:
                    answer = generated_text[-1]['content'].split('/reasoning')[1].replace('\n', '').strip()
                except:
                    answer = generated_text[-1]['content'].replace('\n', '').strip()
        result_data[idx]['output'] = {"answer": answer.strip()}

    return result_data



# Self-Reflection
def generate_with_split_rationale(args, retriever, pipe, result_data):
    logging.info("### Generate with split rationale ###")
    prompts = []
    system_prompt = make_system_prompt()
    topic_keywords = []

    logging.info("Preparing prompts...")
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        topic_keywords.append(topic_keyword)
        context = ""
        if args.retrieve or args.retrieve_adaptively:
            context = retrieve_documents(topic_keyword, question, retriever)
        
        user_prompt = make_prompt_for_only_rationale(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
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

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    prompts = []
    solutions = []
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        instruction = generated_text[1]['content'].split("[지침]\n")[1].split("\n\n")[0]
        question = generated_text[1]['content'].split("[질문]\n")[1].split("[답변]")[0]
        answer = generated_text[-1]['content']
        user_prompt = make_prompt_for_only_answer(
            question=question,
            instruction=instruction,
            answer=answer,
            topic_keyword=topic_keywords[idx],
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompts.append(messages)
        solutions.append(answer)
        
    outputs = pipe(prompts)
    
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        result_data[idx]["input"]["solution"] = solutions[idx].strip()
        result_data[idx]["output"] = {"answer": answer.strip()}

    return result_data





# Consider logits for multiple choice
def generate_for_multiple_choice(args, retriever, pipe, result_data):
    logging.info("### Generate for multiple choice ###")
    prompts = []
    other_indices = []
    system_prompt = make_system_prompt()

    logging.info("Preparing prompts...")
    for idx, item in enumerate(tqdm.tqdm(result_data)):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        context = ""
        if args.retrieve or args.retrieve_adaptively:
            context = retrieve_documents(topic_keyword, question, retriever)

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
                context = retrieve_documents(topic_keyword, question, retriever)
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

    return result_data




def generate_with_verifier(args, retriever, pipe, result_data):
    logging.info("### Generate with verifier ###")
    prompts = []
    system_prompt = make_system_prompt()
    system_prompt_for_verifier = make_system_prompt_for_verifier()
    system_prompt_for_feedback = make_system_prompt_with_feedback()
    
    logging.info("Preparing prompts...")
    topic_keywords = []
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        topic_keywords.append(topic_keyword)
        context = ""
        if args.retrieve or args.retrieve_adaptively:
            logging.info("Retrieving relevant documents...")
            documents = retriever.invoke(question)
            context = format_docs(documents)
        
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
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

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    prompts = []
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        instruction = generated_text[1]['content'].split("[지침]\n")[1].split("\n\n")[0]
        answer = generated_text[-1]['content']
        question = generated_text[1]['content'].split("[질문]\n")[1].split("[답변]")[0]
        verifier_prompt = make_verifier_prompt(instruction=instruction, topic_keyword=topic_keywords[idx], question=question, answer=answer)
        messages = [
            {"role": "system", "content": system_prompt_for_verifier},
            {"role": "user", "content": verifier_prompt}
        ]
        prompts.append(messages)
        
    outputs = pipe(prompts)
    
    regenerate_idx = []
    prompts = []
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        verifier_answer = generated_text[-1]['content']
        if verifier_answer[0] == "예":
            answer = generated_text[-2]['content'].split("[답변]\n")[1].split("이 답변은 질문에 올바르게 대답한 것입니까?\n")[0].replace('\n', '').strip()
            result_data[idx]['output'] = {"answer": answer}
        else:
            regenerate_idx.append(idx)
            instruction = generated_text[1]['content'].split("[질문]\n")[0].split("\n\n")[0]
            answer = generated_text[-2]['content'].split("[답변]\n")[1].split("이 답변은 질문에 올바르게 대답한 것입니까?\n")[0].replace('\n', '').strip()
            question = generated_text[1]['content'].split("[질문]\n")[1].split("[답변]")[0]
            verifier_prompt = make_prompt_with_feedback(instruction=instruction, topic_keyword=topic_keywords[idx], question=question, answer=answer, feedback=verifier_answer[3:])
            messages = [
                {"role": "system", "content": system_prompt_for_feedback},
                {"role": "user", "content": verifier_prompt}
            ]
            prompts.append(messages)

    if len(regenerate_idx) > 0:
        logging.info("Regenerating answers...")
        outputs = pipe(prompts)
        for idx, output in enumerate(tqdm.tqdm(outputs)):
            generated_text = output[0]['generated_text']
            try:
                answer = generated_text[-1]['content'].split("<answer>")[1].split("</answer>")[0].replace('\n', '').strip()
            except:
                answer = generated_text[-1]['content'].replace('\n', '').strip()
            result_data[regenerate_idx[idx]]['output'] = {"answer": answer}
    
    logging.info(f"Number of Regenerated Answers : {len(regenerate_idx)}")
    logging.info(f"Regenerated Answers: {regenerate_idx}")
    return result_data



# Generate Answers
def generate_with_verified_context(args, retriever, pipe, result_data):
    logging.info("### Generate answers ###")
    prompts = []
    system_prompt = make_system_prompt()
    
    logging.info("Preparing prompts...")
    
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]

        context = item['output']['context']
        
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
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

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        # The output from the pipeline is a list with a dictionary
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        result_data[idx]["output"] = {"answer": answer.strip()}

    return result_data



def generate_final(args, pipe, result_data):
    logging.info("### Generate answers ###")
    prompts = []
    system_prompt = make_system_prompt()
    
    logging.info("Preparing prompts...")
    regenerate_idx = []
    for idx, item in tqdm.tqdm(enumerate(result_data)):
        if item['output']['answer'].isdigit():
            pass
        user_prompt = final_answer_prompt_for_MC(
            topic_keyword=item["input"]["topic_keyword"],
            question=item["input"]["question"],
            answer = item["output"]["answer"],
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # pipeline's tokenizer will apply the chat template
        prompts.append(messages)
        regenerate_idx.append(idx)

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        # The output from the pipeline is a list with a dictionary
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        result_data[regenerate_idx[idx]]["input"]["rationale"] = result_data[regenerate_idx[idx]]["output"]["answer"]
        result_data[regenerate_idx[idx]]["output"] = {"answer": answer.strip()}

    return result_data