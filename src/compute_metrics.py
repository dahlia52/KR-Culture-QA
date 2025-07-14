import evaluate
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, EvalPrediction

def compute_metrics(eval_preds: EvalPrediction, eval_dataset: Dataset, tokenizer: AutoTokenizer):
    # 1. Load metrics
    accuracy_metric = evaluate.load("accuracy")
    exact_match_metric = evaluate.load("exact_match")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")
    bleurt_metric = evaluate.load("bleurt", module_type="metric")

    # 2. Decode predictions and labels
    predictions, labels = eval_preds

    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace -100 (token used for masking) with pad_token_id
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 3. Group by question_type
    grouped_examples = {
        '선다형': {'preds': [], 'labels': []},
        '단답형': {'preds': [], 'labels': []},
        '서술형': {'preds': [], 'labels': []},
    }

    for i, item in enumerate(eval_dataset):
        try:
            # Check if 'input' exists and has 'question_type'
            if isinstance(item, dict) and 'input' in item and isinstance(item['input'], dict) and 'question_type' in item['input']:
                q_type = item['input']['question_type']
                if q_type in grouped_examples:
                    grouped_examples[q_type]['preds'].append(decoded_preds[i])
                    grouped_examples[q_type]['labels'].append(decoded_labels[i])
        except Exception as e:
            print(f"Warning: Error processing example {i}: {str(e)}")
            continue

    # 4. Compute metrics for each group
    results = {}
    # '선다형': accuracy
    if grouped_examples['선다형']['preds']:
        acc_results = accuracy_metric.compute(predictions=grouped_examples['선다형']['preds'], references=grouped_examples['선다형']['labels'])
        results['accuracy'] = acc_results['accuracy']

    # '단답형': exact_match with multiple acceptable answers
    if grouped_examples['단답형']['preds']:
        correct = 0
        total = len(grouped_examples['단답형']['labels'])
        
        for pred, ref in zip(grouped_examples['단답형']['preds'], grouped_examples['단답형']['labels']):
            # Split reference by '#' to get all acceptable answers
            acceptable_answers = [ans.strip() for ans in ref.split('#')]
            # Check if prediction matches any of the acceptable answers
            if any(pred.strip() == ans for ans in acceptable_answers):
                correct += 1
        
        results['exact_match'] = correct / total if total > 0 else 0.0

    # '서술형': ROUGE, BERTScore, BLEURT average
    if grouped_examples['서술형']['preds']:
        preds = grouped_examples['서술형']['preds']
        labels = grouped_examples['서술형']['labels']
        
        rouge_results = rouge_metric.compute(predictions=preds, references=labels)
        bertscore_results = bertscore_metric.compute(predictions=preds, references=labels, lang='ko')
        bleurt_results = bleurt_metric.compute(predictions=preds, references=labels)
        
        results['rouge1'] = rouge_results['rouge1']
        results['bertscore_f1'] = np.mean(bertscore_results['f1'])
        results['bleurt'] = np.mean(bleurt_results['scores'])
        
        # 서술형 average score
        descriptive_avg = np.mean([results['rouge1'], results['bertscore_f1'], results['bleurt']])
        results['descriptive_avg_score'] = descriptive_avg

    return results

