#!/bin/sh

echo "Start Inference"

# 실행 명령어
python -m run.test --model_id "jjae/Midm-KCulture-2.0-Base-Instruct" --input "./resource/QA/korean_culture_qa_V1.0_test+.json" --output "./resource/QA/result_0.01.json" --quantize --temperature 0.01 --device "cuda:6"