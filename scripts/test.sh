#!/bin/sh

echo "Start Inference"

# 실행 명령어
python -m run.test --model_id "jjae/Midm-KCulture-2.0-Base-Instruct" --input "./resource/QA/data/korean_culture_qa_V1.0_test+.json" --output "./resource/QA/results/result.json" --quantize --device "cuda"
