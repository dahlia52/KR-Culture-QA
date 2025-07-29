​#!/bin/sh
 
#SBATCH -J inference_verify         
#SBATCH -o src/inference_verify-%j.out   
#SBATCH -p RTX4090         
#SBATCH -t 3-00:00:00            
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBTACH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6

# 환경 설정
echo "Activate conda"
source /home/wisdomjeong/mniconda3/bin/activate kcqa
cd /home/wisdomjeong/KR-Culture-QA
pwd

echo "Start Running"

# 실행 명령어
python src/inference_transformers_by_type_verify.py