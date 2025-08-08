#!/bin/sh

echo "Start Transform Data"

# 실행 명령어
python -m run.merge
python -m run.transform_data --model_id "K-intelligence/Midm-2.0-Base-Instruct" --input "./resource/QA/data/korean_culture_qa_V1.0_total+.json" --output "./resource/QA/data/korean_culture_qa_V1.0_total+_transform.json" --device "cuda"
