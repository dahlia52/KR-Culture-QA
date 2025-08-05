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

    df_category = pd.DataFrame(category_data)[['문화 지식', '문화 실행', '문화 관점']]
    df_category.index = ['train', 'dev', 'test']
    df_category.T.plot(kind='bar', figsize=(8, 6))
    plt.ylabel('Count')
    plt.title('Category Distribution')
    plt.show()
    plt.savefig('./assets/category_distribution.png')
    df_category['총계'] = df_category.sum(axis=1)

    df_question_type = pd.DataFrame(question_type_data)[['선다형', '단답형', '서술형']]
    df_question_type.index = ['train', 'dev', 'test']
    df_question_type.T.plot(kind='bar', figsize=(8, 6))
    plt.ylabel('Count')
    plt.title('Question Type Distribution')
    plt.show()
    plt.savefig('./assets/question_type_distribution.png')
    df_question_type['총계'] = df_question_type.sum(axis=1)

    df_domain = pd.DataFrame(domain_data)[['일상생활', '예술', '풍습/문화유산', '과학기술', '지리', '교육', '가치관', '사회', '역사', '정치/경제']]
    df_domain.index = ['train', 'dev', 'test']
    df_domain.T.plot(kind='bar', figsize=(12, 9))
    plt.ylabel('Count')
    plt.title('Domain Distribution')
    plt.show()
    plt.savefig('./assets/domain_distribution.png')
    df_domain['총계'] = df_domain.sum(axis=1)
    
    df_question_type.to_csv('./assets/question_type_distribution.csv', encoding = 'cp949')
    print(f"Question type distribution saved to ./assets/question_type_distribution.csv")

    df_domain.to_csv('./assets/domain_distribution.csv', encoding = 'cp949')
    print(f"Domain distribution saved to ./assets/domain_distribution.csv")

    df_category.to_csv('./assets/category_distribution.csv', encoding = 'cp949')
    print(f"Category distribution saved to ./assets/category_distribution.csv")


# Question Type별 선지 개수
#def question_distribution():




def generate_word_cloud(): # texts, output_path, max_words=100, width=800, height=400, background_color='white'
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Initialize Okt for Korean tokenization
    okt = Okt()
    
    # Tokenize and extract nouns
    tokens = okt.nouns(combined_text)
    tokenized_text = ' '.join(tokens)
    
    # Generate word cloud
    wordcloud = WordCloud(
        font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # Adjust path if needed
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        colormap='viridis'
    ).generate(tokenized_text)
    
    # Save the word cloud
    wordcloud.to_file(output_path)
    print(f"Word cloud saved to {output_path}")

def extract_questions(data):
    """Extract questions from the dataset."""
    questions = []
    for item in data:
        if 'input' in item and 'question' in item['input']:
            questions.append(item['input']['question'])
    return questions

def generate_word_clouds_for_splits(files, output_dir):
    """Generate word clouds for each split in the dataset."""
    for split, file_path in files.items():
        print(f"Generating word cloud for {split} data...")
        try:
            data = load_dataset(file_path)
            questions = extract_questions(data)
            
            if questions:
                output_path = os.path.join(output_dir, f'wordcloud_{split}.png')
                generate_word_cloud(questions, output_path)
                print(f"  - Generated word cloud with {len(questions)} questions")
            else:
                print(f"  - No questions found in {split} data")
                
        except Exception as e:
            print(f"Error generating word cloud for {file_path}: {e}")

def main():
    # File paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = Path(current_dir) / 'resource' / 'QA'
    files = {
        'train': base_dir / 'korean_culture_qa_V1.0_train+.json',
        'dev': base_dir / 'korean_culture_qa_V1.0_dev+.json',
        'test': base_dir / 'korean_culture_qa_V1.0_test+.json'
    }
    data_distribution(files)


if __name__ == "__main__":
    main()