#!/bin/sh

echo "Start Training"

# 실행 명령어
python -m run.train --model_id "K-intelligence/Midm-2.0-Base-Instruct" --train_data_path "./resource/QA/korean_culture_qa_V1.0_total+_transform.json" --output "./models/K-intelligence-Midm-2.0-Base-Instruct-fine-tuned" --epochs 3 --gpu_ids "0,1"