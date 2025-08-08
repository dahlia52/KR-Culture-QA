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
