#!/bin/sh

echo "Start Remaking Data"

# 실행 명령어
python -m run.merge
python -m run.remake_data --model_id "K-intelligence/Midm-2.0-Base-Instruct" --input "./resource/QA/korean_culture_qa_V1.0_total+.json" --output "./resource/QA/korean_culture_qa_V1.0_total+_remake.json" --device "cuda"
