<div align='center'>
   <h1>2025년 국립국어원 AI 말평 경진 대회<br>- 한국문화 질의응답 (나 유형)</h1>
</div>

<div align="center">
    <p>국립국어원 인공지능(AI)말평 - <a href="https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=181&clCd=ING_TASK&subMenuId=sub01" target="_blank">2025년 한국문화 질의응답 (나 유형)</a></p>
</div>
<br>


# 디렉토리 구조

```bash
KR-Culture-QA
├── resource # 학습에 필요한 리소스들을 보관하는 디렉토리
│   ├── QA # 학습, 평가 데이터셋을 보관하는 디렉토리
│   │   └── korean_culture_qa_V1.0_train+.json
│   │   └── korean_culture_qa_V1.0_dev+.json
│   │   └── korean_culture_qa_V1.0_test+.json
│   │   └── korean_culture_qa_V1.0_total+.json
│   │   └── korean_culture_qa_V1.0_total+_remake.json
│   └── retrieval_docs # 검색을 위해 데이터, DB를 구축하는 디렉토리
│       └── download_rag_data.py 
│       └── download_chromadb.py 
├── run # 실행 가능한 python 스크립트를 보관하는 디렉토리
│   ├── merge.py
│   ├── remake_data.py
│   ├── EDA.py
│   ├── train.py
│   └── test.py
├── scripts # 학습 및 추론을 실행하기 위한 bash 스크립트를 보관하는 디렉토리
│   ├── remake.sh # 학습 데이터를 재구성하기 위한 bash 스크립
│   ├── train.sh # 학습을 실행하기 위한 bash 스크립트
│   └── test.sh # 추론을 실행하기 위한 bash 스크립트
└── src # 학습 및 추론에 사용될 함수들을 보관하는 디렉토리
    └── data_io.py
    └── load_model.py
    └── retrieve.py
    └── make_prompt.py
    └── generate.py
    └── postprocess.py
```

# 실행 방법

## Requirements
코드 실행을 위해 아래와 같은 환경이 필요합니다.
- Ubuntu 22.04.4 LTS
- Python 3.12.9
- Miniconda 24.11.3
- git


### Miniconda 설치
```bash
$ cd ~ # 설치 파일을 다운로드할 경로로 이동 (to home directory)
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # Miniconda 설치 파일 다운로드
$ bash Miniconda3-latest-Linux-x86_64.sh # 설치 파일 실행
$ export PATH=~/miniconda3/bin:$PATH # 환경 변수 설정
$ source ~/.bashrc # Anaconda 설치 후 bash shell 환경 재설정
$ conda init # conda 초기화
$ conda --version # conda 버전 확인
```

## 환경 설정

### 개발 환경 설정
```bash
$ git clone https://github.com/dahlia52/KR-Culture-QA.git
$ cd KR-Culture-QA
$ conda create -n krqa python=3.12.9
$ conda activate krqa
$ pip install -r requirements.txt
```

### 한글 폰트 설치 (EDA 시 필요)
```bash
$ curl -o nanumfont.zip http://cdn.naver.com/naver/NanumFont/fontfiles/NanumFont_TTF_ALL.zip
$ sudo unzip -d /usr/share/fonts/nanum nanumfont.zip
$ sudo fc-cache -f -v
$ fc-list | grep Nanum
$ rm ~/.cache/matplotlib/fontlist*
```

## 데이터셋 준비
- 학습 데이터셋 증강
```bash
# train.json과 dev.json을 합친 후 선다형 문제를 서술형 문제로 변경
sh scripts/transform.sh
```
- 
```bash
# 한국어 위키피디아 데이터 내려받기
python –m resource/retrieval_docs/download_rag_data.py
```

```bash
# ChromaDB 내려받기
python resource/retrieval_docs/download_chromadb.py
```

## EDA (Exploratory Data Analysis)
데이터셋을 분석하기 위해 아래 명령어를 실행합니다.

```bash
$ python -m run/EDA.py
```

## 학습 (Train)
```bash
$ sh scripts/train.sh
```

## 추론 (Inference)
```bash
$ sh scripts/test.sh
```
