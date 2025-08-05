import tqdm
from src.make_prompt import *
from src.retrieve import *
from src.postprocess import postprocess_data
import logging


# Generate Answers
def generate(args, pipe, result_data, contexts):
    logging.info("### Generate answers ###")
    prompts = []
    questions = []
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
            context=contexts[idx] if contexts else "",
            question=question,
            fewshot=True
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # pipeline's tokenizer will apply the chat template
        prompts.append(messages)
        questions.append(question)

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        answer = postprocess_data(item["input"]["question_type"], questions[idx], answer)
        result_data[idx]["output"] = {"answer": answer.strip()}

    return result_data


# Generate Answers
def generate_with_rationale(args, pipe, result_data, contexts):
    logging.info("### Generate answers ###")
    prompts = []
    system_prompt = make_system_prompt()
    
    logging.info("Preparing prompts...")
    
    for idx, item in tqdm.tqdm(enumerate(result_data)):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        user_prompt = make_prompt_for_rationale(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
            context=contexts[idx] if contexts else "",
            question=question,
            fewshot=True
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
        try:
            answer = answer.replace("assistant", "")
            answer = answer.replace("ass", "")
            answer = answer.replace("\u0000", "")
            if '답변:' in answer:
                answer = answer.split('답변:')[1].strip()
        except:
            pass
        initial_answer = answer.strip()
        try:
            answer = answer.split("<answer>")[1].split("</answer>")[0].replace('\n', '').strip()
            
        except:
            try:
                answer = answer.split("<answer>")[1].replace('\n', '').strip()
            except:
                try:
                    answer = answer.split('/reasoning>')[1].replace('\n', '').strip()
                except:
                    answer = answer.replace('\n', '').strip()

        if not answer.isdigit():
            answer = initial_answer
        result_data[idx]["output"] = {"answer": answer.strip()}
        logging.info(f"Question: {question}")
        logging.info(f"Initial answer: {initial_answer}")

    return result_data



def regenerate(pipe, result_data):
    logging.info("### Generate answers ###")
    prompts = []
    questions = []
    system_prompt = make_system_prompt()
    
    logging.info("Preparing prompts...")
    regenerate_idx = []
    for idx, item in tqdm.tqdm(enumerate(result_data)):
        if item['output']['answer'].isdigit():
            continue
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
        questions.append(item["input"]["question"])
        regenerate_idx.append(idx)

    logging.info("Generating answers in batch...")
    if len(prompts) == 0:
        return result_data
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        # The output from the pipeline is a list with a dictionary
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        answer = postprocess_data(item["input"]["question_type"], questions[idx], answer)
        result_data[regenerate_idx[idx]]["output"] = {"answer": answer.strip()}

    return result_data