import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from matplotlib import rc
from konlpy.tag import Okt
from src.data_io import load_dataset
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
import umap

# Set font family to NanumGothic
rc('font', family='NanumGothic')
 
# Category, Question Type, Domain Distribution
def data_distribution(files):
    question_type_data = []
    domain_data = []
    category_data = []
    
    for split, file_path in files.items():
        data = load_dataset(file_path)
        category = [entry['input']['category'] for entry in data]
        question_types = [entry['input']['question_type'] for entry in data]
        domains = [entry['input']['domain'] for entry in data]

        question_type_count = Counter(question_types)
        domain_count = Counter(domains)
        category_count = Counter(category)

        question_type_data.append(question_type_count)
        domain_data.append(domain_count)
        category_data.append(category_count)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Distribution of Question Type
    df_question_type = pd.DataFrame(question_type_data)[['선다형', '단답형', '서술형']]
    df_question_type.index = ['train', 'dev', 'test']
    df_question_type['총계'] = df_question_type.sum(axis=1)

    df_question_type_plot = df_question_type.drop(columns='총계').T
    df_question_type_plot.plot(kind='bar', ax=axes[0])
    axes[0].set_title('질문 유형 분포포', fontweight='bold')
    axes[0].set_ylabel('Count')

    # Distribution of Domain
    df_domain = pd.DataFrame(domain_data)[['일상생활', '예술', '풍습/문화유산', '과학기술', '지리', '교육', '가치관', '사회', '역사', '정치/경제']]
    df_domain.index = ['train', 'dev', 'test']
    df_domain['총계'] = df_domain.sum(axis=1)

    df_domain_plot = df_domain.drop(columns='총계').T
    df_domain_plot.plot(kind='bar', ax=axes[1])
    axes[1].set_title('도메인 분포', fontweight='bold')
    axes[1].set_ylabel('Count')

    # Distribution of Category
    df_category = pd.DataFrame(category_data)[['문화 지식', '문화 실행', '문화 관점']]
    df_category.index = ['train', 'dev', 'test']
    df_category['총계'] = df_category.sum(axis=1)

    df_category_plot = df_category.drop(columns='총계').T
    df_category_plot.plot(kind='bar', ax=axes[2])
    axes[2].set_title('카테고리 분포', fontweight='bold')
    axes[2].set_ylabel('Count')

    # Save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('./assets/dataset_distribution.png')
    plt.show()
    
    df_question_type.to_csv('./assets/question_type_distribution.csv', encoding = 'cp949')
    print(f"Question type distribution saved to ./assets/question_type_distribution.csv")

    df_domain.to_csv('./assets/domain_distribution.csv', encoding = 'cp949')
    print(f"Domain distribution saved to ./assets/domain_distribution.csv")

    df_category.to_csv('./assets/category_distribution.csv', encoding = 'cp949')
    print(f"Category distribution saved to ./assets/category_distribution.csv")



def create_circle_mask(diameter=600):
    """원형 마스크를 numpy array로 생성"""
    x, y = np.ogrid[:diameter, :diameter]
    center = diameter / 2
    mask = (x - center) ** 2 + (y - center) ** 2 > (center ** 2)
    mask = 255 * mask.astype(int)  # 255: 흰색(워드클라우드 영역 아님), 0: 검은색(워드클라우드 가능)
    return mask
    
# Word Cloud
def generate_word_cloud(files):
    question_data = []
    split_data = []
    for split, file_path in files.items():
        data = load_dataset(file_path)
        questions = [entry['input']['question'] for entry in data]
        question_data.append(questions)
        split_data.append(split)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))


    for idx, questions in enumerate(question_data):
        combined_text = ' '.join(questions)
        
        # Initialize Okt for Korean tokenization
        okt = Okt()
        
        # Tokenize and extract nouns
        tokens = okt.nouns(combined_text)
        tokenized_text = ' '.join(tokens)
        
        # Generate word cloud
        STOPWORDS2 = set([
            # 조사/의존명사/추상명사
            '것', '수', '때', '등', '무엇', '왜', '때', '중', '후', '전', '위', '내', '곳', '속', '중요', '부분'
            # 의문/기본 어휘
            '무엇', '어떤', '어떻게', '누구', '왜', '어디', '언제', '대한', '대한민국', '한국','하나요','위해','서술','모두','설명','다음','이름','주로','대해'
            ])
        stop_words = STOPWORDS.union(STOPWORDS2)
        wordcloud = WordCloud(
            font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # Adjust path if needed
            width=800,
            height=600,
            background_color='white',
            max_words=200,
            colormap='viridis',
            mask=create_circle_mask(),
            stopwords=stop_words
        ).generate(tokenized_text)

        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(split_data[idx].capitalize(), fontsize=20, fontweight='bold')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./assets/wordcloud.png')
    print(f"Word cloud saved to ./assets/wordcloud.png")


# Answer Distribution
def question_length_histogram(files):
    # 질문 길이 데이터 초기화
    lengths = {
        '선다형': {'train': [], 'dev': [], 'test': []},
        '단답형': {'train': [], 'dev': [], 'test': []},
        '서술형': {'train': [], 'dev': [], 'test': []}
    }

    for split, file_path in files.items():
        data = load_dataset(file_path)
        for entry in data:
            q_type = entry['input']['question_type']
            question = entry['input']['question']
            length = len(question)
            lengths[q_type][split].append(length)

    # 시각화: 유형별 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    question_types = ['선다형', '단답형', '서술형']
    colors = {'train': 'skyblue', 'dev': 'orange', 'test': 'green'}

    for idx, q_type in enumerate(question_types):
        ax = axes[idx]
        for split in ['train', 'dev', 'test']:
            ax.hist(lengths[q_type][split], bins=20, alpha=0.6, label=split, color=colors[split])
        ax.set_title(f'{q_type} 질문 길이 분포', fontweight='bold')
        ax.set_xlabel('Question Length')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig('./assets/question_length_distribution.png')
    plt.show()

    

def main():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = Path(current_dir) / 'resource' / 'QA' / 'data'
    files = {
        'train': base_dir / 'korean_culture_qa_V1.0_train+.json',
        'dev': base_dir / 'korean_culture_qa_V1.0_dev+.json',
        'test': base_dir / 'korean_culture_qa_V1.0_test+.json'
    }
    os.makedirs('./assets', exist_ok=True)
    data_distribution(files)
    question_length_histogram(files)
    generate_word_cloud(files)

if __name__ == "__main__":
    main()