#!/usr/bin/env python
# coding: utf-8

# ## 상품 데이터 분석 및 전처리

# In[2]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from tqdm import tqdm
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import platform
import re


# In[3]:


# 시각화 전에 한글 폰트 설정 추가
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정 함수
def set_korean_font():
    system = platform.system()
    
    if system == 'Windows':
        plt.rc('font', family='Malgun Gothic')  # 윈도우
    elif system == 'Darwin':
        plt.rc('font', family='AppleGothic')    # macOS
    else:  # Linux
        try:
            # 나눔 폰트 등 한글 폰트 찾기 시도
            font_list = [f for f in fm.findSystemFonts() if 'Nanum' in f]
            if font_list:
                font_path = font_list[0]
                font_prop = fm.FontProperties(fname=font_path)
                plt.rc('font', family=font_prop.get_name())
            else:
                print("한글 폰트를 찾을 수 없습니다. 카테고리명이 깨질 수 있습니다.")
        except:
            print("한글 폰트 설정 중 오류가 발생했습니다.")
    
    # 음수 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


# In[4]:


# 1. 데이터 로드
def load_data(file_path):
    """
    엑셀 또는 CSV 파일에서 상품 데이터를 로드합니다.
    """
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # 엑셀 파일 처리
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            print(f"openpyxl 엔진으로 로드 실패: {e}. xlrd 엔진 시도 중...")
            df = pd.read_excel(file_path, engine='xlrd')
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. .xlsx, .xls 또는 .csv 파일을 사용해주세요.")
    
    print(f"데이터 로드 완료: {len(df)} 개의 상품 데이터")
    return df


# In[5]:


# 2. 데이터 개요 분석
def analyze_data(df):
    """
    데이터 개요를 분석하고 출력합니다.
    """
    print("\n===== 데이터 개요 =====")
    print(f"상품 개수: {len(df)}")
    print(f"컬럼 개수: {len(df.columns)}")
    print("\n===== 컬럼 정보 =====")
    print(df.info())
    
    print("\n===== 결측치 분석 =====")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        '결측치 개수': missing_data,
        '결측치 비율(%)': missing_percent
    })
    print(missing_info[missing_info['결측치 개수'] > 0].sort_values('결측치 개수', ascending=False))
    
    print("\n===== 카테고리 분포 =====")
    if '카테고리명' in df.columns:
        category_counts = df['카테고리명'].value_counts().head(10)
        print(category_counts)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_counts.values, y=category_counts.index)
        plt.title('상위 10개 카테고리 분포')
        plt.xlabel('상품 개수')
        plt.tight_layout()
        plt.savefig('category_distribution.png')
    
    return missing_info


# In[6]:


# 2. 데이터 개요 분석 - 한국어
def analyze_data(df):
    """
    데이터 개요를 분석하고 출력합니다.
    """
    print("\n===== 데이터 개요 =====")
    print(f"상품 개수: {len(df)}")
    print(f"컬럼 개수: {len(df.columns)}")
    print("\n===== 컬럼 정보 =====")
    print(df.info())
    
    print("\n===== 결측치 분석 =====")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        '결측치 개수': missing_data,
        '결측치 비율(%)': missing_percent
    })
    print(missing_info[missing_info['결측치 개수'] > 0].sort_values('결측치 개수', ascending=False))
    
    print("\n===== 카테고리 분포 =====")
    if '카테고리명' in df.columns:
        category_counts = df['카테고리명'].value_counts().head(10)
        print(category_counts)
        
        # 방법 1: 영어로 카테고리 표시하기
        plt.figure(figsize=(12, 6))
        # 카테고리 이름을 인덱스 번호로 대체
        category_indices = [f"Category {i+1}" for i in range(len(category_counts))]
        
        # 실제 카테고리 이름과 인덱스 번호 매핑 출력
        print("\n카테고리 매핑:")
        for i, (cat, count) in enumerate(category_counts.items()):
            print(f"Category {i+1}: {cat} ({count} items)")
        
        # 그래프 생성
        sns.barplot(x=category_counts.values, y=category_indices)
        plt.title('Top 10 Categories Distribution')
        plt.xlabel('Number of Products')
        plt.tight_layout()
        plt.savefig('category_distribution.png')
    
    return missing_info


# In[7]:


# 3. 텍스트 전처리
def preprocess_text(text):
    """
    텍스트를 전처리합니다.
    """
    if pd.isna(text):
        return ""
    
    # 텍스트 타입 확인
    if not isinstance(text, str):
        text = str(text)
    
    # 특수문자 제거 (단, 한글, 영문, 숫자, 일부 특수문자는 유지)
    text = re.sub(r'[^\wㄱ-ㅎㅏ-ㅣ가-힣\s\,\.\-\/]', ' ', text)
    
    # 여러 공백을 하나의 공백으로 대체
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# In[8]:


# 4. 데이터 전처리
def preprocess_data(df):
    """
    데이터를 전처리하고 추천 시스템에 필요한 필드를 통합합니다.
    """
    # 복사본 생성
    processed_df = df.copy()
    
    # 주요 필드 전처리
    text_columns = ['원본상품명', '카테고리명', '키워드', '정보고시 항목정보', '모델명', '제작/수입사']
    
    for col in text_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].apply(preprocess_text)
    
    # 결측치 처리
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            processed_df[col] = processed_df[col].fillna('')
        else:
            processed_df[col] = processed_df[col].fillna(0)
    
    # 상품 설명 텍스트 통합
    processed_df['통합_텍스트'] = (
        processed_df['원본상품명'] + ' ' + 
        processed_df['카테고리명'] + ' ' + 
        processed_df['키워드'] + ' ' + 
        processed_df['모델명'].astype(str) + ' ' + 
        processed_df['정보고시 항목정보'].astype(str)
    )
    
    # 가격 정보 정규화
    if '오너클랜판매가' in processed_df.columns:
        price_max = processed_df['오너클랜판매가'].max()
        price_min = processed_df['오너클랜판매가'].min()
        processed_df['가격_정규화'] = (processed_df['오너클랜판매가'] - price_min) / (price_max - price_min)
    
    return processed_df


# In[9]:


# 5. 데이터 저장
def save_processed_data(df, output_path):
    """
    전처리된 데이터를 저장합니다.
    """
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"전처리된 데이터가 {output_path}에 저장되었습니다.")


# In[10]:


input_file = "narosu_db_final.xlsx"  # 실제 엑셀 파일 경로로 변경하세요
output_file = "narosu_db_final.csv"


# In[11]:


# 1. 데이터 로드
df = load_data(input_file)


# In[12]:


# 2. 데이터 분석
missing_info = analyze_data(df)


# In[13]:


# 3. 데이터 전처리
processed_df = preprocess_data(df)


# In[14]:


# 4. 전처리된 데이터 저장
save_processed_data(processed_df, output_file)


# ## 임베딩 모델 및 벡터 저장소 구축

# In[15]:


import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from openai import OpenAI


# In[16]:


OPENAI_API_KEY = "secret"


# In[17]:


# 한글 폰트 설정 함수
def set_korean_font():
    import platform
    system = platform.system()
    
    if system == 'Windows':
        plt.rc('font', family='Malgun Gothic')  # 윈도우
    elif system == 'Darwin':
        plt.rc('font', family='AppleGothic')    # macOS
    else:  # Linux
        try:
            # 나눔 폰트 등 한글 폰트 찾기 시도
            import matplotlib.font_manager as fm
            font_list = [f for f in fm.findSystemFonts() if 'Nanum' in f]
            if font_list:
                font_path = font_list[0]
                font_prop = fm.FontProperties(fname=font_path)
                plt.rc('font', family=font_prop.get_name())
            else:
                print("한글 폰트를 찾을 수 없습니다. 카테고리명이 깨질 수 있습니다.")
        except:
            print("한글 폰트 설정 중 오류가 발생했습니다.")
    
    # 음수 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


# 1. 데이터 로드
def load_data(file_path):
    """
    엑셀 또는 CSV 파일에서 상품 데이터를 로드합니다.
    """
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # 엑셀 파일 처리
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            print(f"openpyxl 엔진으로 로드 실패: {e}. xlrd 엔진 시도 중...")
            df = pd.read_excel(file_path, engine='xlrd')
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. .xlsx, .xls 또는 .csv 파일을 사용해주세요.")
    
    print(f"데이터 로드 완료: {len(df)} 개의 상품 데이터")
    return df


# 2. 데이터 개요 분석 - 한국어
def analyze_data(df):
    """
    데이터 개요를 분석하고 출력합니다.
    """
    print("\n===== 데이터 개요 =====")
    print(f"상품 개수: {len(df)}")
    print(f"컬럼 개수: {len(df.columns)}")
    print("\n===== 컬럼 정보 =====")
    print(df.info())
    
    print("\n===== 결측치 분석 =====")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        '결측치 개수': missing_data,
        '결측치 비율(%)': missing_percent
    })
    print(missing_info[missing_info['결측치 개수'] > 0].sort_values('결측치 개수', ascending=False))
    
    print("\n===== 카테고리 분포 =====")
    if '카테고리명' in df.columns:
        category_counts = df['카테고리명'].value_counts().head(10)
        print(category_counts)
        
        # 방법 1: 영어로 카테고리 표시하기
        plt.figure(figsize=(12, 6))
        # 카테고리 이름을 인덱스 번호로 대체
        category_indices = [f"Category {i+1}" for i in range(len(category_counts))]
        
        # 실제 카테고리 이름과 인덱스 번호 매핑 출력
        print("\n카테고리 매핑:")
        for i, (cat, count) in enumerate(category_counts.items()):
            print(f"Category {i+1}: {cat} ({count} items)")
        
        # 그래프 생성
        sns.barplot(x=category_counts.values, y=category_indices)
        plt.title('Top 10 Categories Distribution')
        plt.xlabel('Number of Products')
        plt.tight_layout()
        plt.savefig('category_distribution.png')
    
    return missing_info


# 3. 텍스트 전처리
def preprocess_text(text):
    """
    텍스트를 전처리합니다.
    """
    if pd.isna(text):
        return ""
    
    # 텍스트 타입 확인
    if not isinstance(text, str):
        text = str(text)
    
    # 특수문자 제거 (단, 한글, 영문, 숫자, 일부 특수문자는 유지)
    text = re.sub(r'[^\wㄱ-ㅎㅏ-ㅣ가-힣\s\,\.\-\/]', ' ', text)
    
    # 여러 공백을 하나의 공백으로 대체
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# 4. 데이터 전처리
def preprocess_data(df):
    """
    데이터를 전처리하고 추천 시스템에 필요한 필드를 통합합니다.
    """
    # 복사본 생성
    processed_df = df.copy()
    
    # 주요 필드 전처리
    text_columns = ['원본상품명', '카테고리명', '키워드', '정보고시 항목정보', '모델명', '제작/수입사']
    
    for col in text_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].apply(preprocess_text)
    
    # 결측치 처리
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            processed_df[col] = processed_df[col].fillna('')
        else:
            processed_df[col] = processed_df[col].fillna(0)
    
    # 상품 설명 텍스트 통합
    processed_df['통합_텍스트'] = (
        processed_df['원본상품명'] + ' ' + 
        processed_df['카테고리명'] + ' ' + 
        processed_df['키워드'] + ' ' + 
        processed_df['모델명'].astype(str) + ' ' + 
        processed_df['정보고시 항목정보'].astype(str)
    )
    
    # 가격 정보 정규화
    if '오너클랜판매가' in processed_df.columns:
        price_max = processed_df['오너클랜판매가'].max()
        price_min = processed_df['오너클랜판매가'].min()
        processed_df['가격_정규화'] = (processed_df['오너클랜판매가'] - price_min) / (price_max - price_min)
    
    return processed_df


# 5. 데이터 저장
def save_processed_data(df, output_path):
    """
    전처리된 데이터를 저장합니다.
    """
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"전처리된 데이터가 {output_path}에 저장되었습니다.")



# In[18]:


# 1. OpenAI 임베딩 모델 설정 - 수정된 버전
def get_embedding_model():
    """
    OpenAI 임베딩 모델을 설정합니다.
    API 키를 직접 지정합니다.
    """
    # 직접 API 키 지정
    client = OpenAI(api_key=OPENAI_API_KEY)
    # text-embedding-ada-002 모델 사용
    model_name = "text-embedding-ada-002"
    return client, model_name


# 2. 텍스트를 임베딩 벡터로 변환 - 수정된 버전
def get_embedding(text, model="text-embedding-ada-002"):
    """
    OpenAI API를 사용하여 텍스트의 임베딩 벡터를 생성
    API 키를 직접 지정합니다.
    """
    # 클라이언트 초기화 (API 키 직접 지정)
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None


# 3. 임베딩 생성 (배치 처리) - 수정된 버전
def create_embeddings_batch(df, text_column, model_name, batch_size=100):
    """
    데이터프레임의 텍스트 컬럼을 배치로 처리하여 임베딩을 생성
    client 인자 제거하고 내부에서 생성합니다.
    """
    embeddings = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(df), batch_size), desc="임베딩 생성 중", total=total_batches):
        batch_df = df.iloc[i:i+batch_size]
        batch_texts = batch_df[text_column].tolist()
        
        # 텍스트 전처리 및 임베딩 생성
        batch_embeddings = []
        for text in batch_texts:
            if pd.isna(text) or text == "":
                # 빈 텍스트는 0으로 채운 임베딩 생성
                batch_embeddings.append([0.0] * 1536)  # text-embedding-ada-002는 1536 차원
            else:
                embedding = get_embedding(text, model_name)
                if embedding:
                    batch_embeddings.append(embedding)
                else:
                    batch_embeddings.append([0.0] * 1536)
        
        embeddings.extend(batch_embeddings)
        
        # API 속도 제한 방지를 위해 잠시 대기
        time.sleep(0.5)
    
    return embeddings


# 4. Qdrant 벡터 저장소 설정 - 필요한 경우
def setup_qdrant(collection_name, vector_size=1536):
    """
    Qdrant 벡터 저장소를 설정합니다.
    """
    try:
        # Qdrant 관련 모듈 임포트 (필요한 경우)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # Qdrant 클라이언트 생성 (로컬 모드)
        client = QdrantClient(":memory:")  # 메모리 모드 (실제 배포시 서버 URL 사용)
        
        # 컬렉션이 이미 존재하는지 확인
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        # 컬렉션이 없으면 생성
        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Qdrant 컬렉션 '{collection_name}'이(가) 생성되었습니다.")
        else:
            print(f"Qdrant 컬렉션 '{collection_name}'이(가) 이미 존재합니다.")
        
        return client
    except Exception as e:
        print(f"Qdrant 설정 중 오류 발생: {e}")
        return None


# 5. 임베딩을 Qdrant에 저장 - Qdrant 관련 모듈 수정
def store_embeddings_in_qdrant(collection_name, df, embeddings):
    """
    생성된 임베딩을 Qdrant 벡터 저장소에 저장
    client 인자 제거하고 내부에서 생성
    """
    try:
        # Qdrant 관련 모듈 임포트
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        
        # 클라이언트 생성
        client = QdrantClient(":memory:")  # 메모리 모드
        
        points = []
        for i, embedding in enumerate(embeddings):
            if embedding:
                # 메타데이터 구성
                metadata = {
                    "상품코드": df.iloc[i].get("상품코드", ""),
                    "원본상품명": df.iloc[i].get("원본상품명", ""),
                    "카테고리명": df.iloc[i].get("카테고리명", ""),
                    "가격": int(df.iloc[i].get("오너클랜판매가", 0)),
                    "키워드": df.iloc[i].get("키워드", ""),
                    "이미지URL": df.iloc[i].get("이미지대", "")
                }
                
                point = PointStruct(
                    id=i,
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)
        
        # 배치 업로드
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"{len(points)}개의 임베딩이 Qdrant에 성공적으로 저장되었습니다.")
        return True
    except Exception as e:
        print(f"Qdrant에 저장 중 오류 발생: {e}")
        return False


# 6. 임베딩을 파일로 저장 (백업용)
def save_embeddings_to_file(embeddings, output_path):
    """
    생성된 임베딩을 파일로 저장합니다. (백업용)
    """
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"임베딩이 {output_path}에 저장되었습니다.")
    return True


# 7. 샘플링 함수 (대용량 데이터셋 처리용)
def sample_data(df, sample_size=10000, stratify_column='카테고리명'):
    """
    대용량 데이터셋에서 카테고리 비율을 유지하며 샘플링합니다.
    """
    if len(df) <= sample_size:
        return df
    
    if stratify_column in df.columns:
        # 카테고리별 비율을 유지하며 샘플링
        try:
            sampled_df = pd.DataFrame()
            categories = df[stratify_column].unique()
            
            for category in categories:
                category_df = df[df[stratify_column] == category]
                category_sample_size = int(len(category_df) / len(df) * sample_size)
                
                if category_sample_size > 0:
                    category_sample = category_df.sample(min(category_sample_size, len(category_df)))
                    sampled_df = pd.concat([sampled_df, category_sample])
            
            # 샘플 크기 조정
            if len(sampled_df) > sample_size:
                sampled_df = sampled_df.sample(sample_size)
            
            return sampled_df
        except Exception as e:
            print(f"계층적 샘플링 중 오류 발생: {e}. 단순 랜덤 샘플링을 사용합니다.")
    
    # 단순 랜덤 샘플링
    return df.sample(sample_size)


# API 키 검증 함수 (추가)
def validate_api_key():
    """
    OpenAI API 키를 검증합니다.
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input="API 키 검증용 텍스트",
            model="text-embedding-ada-002"
        )
        print("✅ API 키가 유효합니다.")
        return True
    except Exception as e:
        print(f"❌ API 키 검증 실패: {e}")
        return False


# In[19]:


if not validate_api_key():
    print("API 키가 유효하지 않습니다. 프로그램을 종료합니다.")
    exit(1)


# In[20]:


# 1. 전처리된 데이터 로드
input_file = "narosu_db_final.csv"  # 전처리된 파일 경로
df = pd.read_csv(input_file)
print(f"전처리된 데이터 로드 완료: {len(df)}개의 상품")


# In[21]:


# 2. 대용량 데이터를 위한 샘플링 (테스트용, 실제 구현시 전체 데이터 사용) 
sample_size = 1000000  # 샘플 데이터 전체로 사용
sampled_df = sample_data(df, sample_size)
print(f"샘플링된 데이터: {len(sampled_df)}개의 상품")


# In[22]:


import logging
# httpx 로깅 비활성화 (WARNING 이상 레벨만 표시)
logging.getLogger("httpx").setLevel(logging.WARNING)


# In[23]:


import concurrent.futures
import time
from tqdm import tqdm
from openai import OpenAI

def get_embedding_with_retry(text, model_name, client, max_retries=3):
    """단일 텍스트에 대한 임베딩을 가져오는 함수 (재시도 로직 포함)"""
    if pd.isna(text) or text == "":
        return [0.0] * 1536  # text-embedding-ada-002는 1536 차원
    
    text = text.replace("\n", " ")
    retries = 0
    
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                input=text,
                model=model_name
            )
            return response.data[0].embedding
        except Exception as e:
            retries += 1
            if retries < max_retries:
                wait_time = 2 ** retries  # 지수 백오프
                print(f"임베딩 오류 (재시도 {retries}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                print(f"최대 재시도 횟수 초과. 임베딩 실패: {e}")
                return [0.0] * 1536  # 오류 시 0 벡터 반환

import concurrent.futures
import time
import os
import pickle
from tqdm import tqdm
from openai import OpenAI

def create_embeddings_parallel_with_checkpoints(
    df, 
    text_column, 
    model_name, 
    client, 
    output_dir="embeddings_checkpoints",
    checkpoint_interval=1000,  # 몇 개 항목마다 체크포인트 저장할지
    start_index=0,  # 시작 인덱스 (이어서 처리할 경우)
    batch_size=100, 
    max_workers=10
):
    """체크포인트 기능이 있는 병렬 임베딩 생성 함수"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 체크포인트 파일 경로
    checkpoint_file = os.path.join(output_dir, "embeddings_checkpoint.pkl")
    
    # 이미 생성된 임베딩이 있는지 확인
    existing_embeddings = []
    last_processed_index = start_index - 1  # 마지막으로 처리된 인덱스
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                existing_embeddings = checkpoint_data.get('embeddings', [])
                last_processed_index = checkpoint_data.get('last_index', -1)
                
                print(f"체크포인트에서 {len(existing_embeddings)}개 임베딩 로드됨")
                print(f"마지막 처리 인덱스: {last_processed_index}")
                
                # 시작 인덱스 조정
                start_index = max(start_index, last_processed_index + 1)
        except Exception as e:
            print(f"체크포인트 로드 오류: {e}")
    
    # 처리할 데이터 추출
    if start_index >= len(df):
        print("이미 모든 데이터가 처리되었습니다.")
        return existing_embeddings
    
    texts = df[text_column].iloc[start_index:].tolist()
    
    # 결과 저장할 리스트 (기존 임베딩에 새로운 임베딩 추가)
    embeddings = list(existing_embeddings)
    
    # 진행 상황을 표시할 tqdm 객체 생성
    pbar = tqdm(total=len(texts), desc="임베딩 생성 중")
    
    # 이전에 처리된 항목을 표시
    if existing_embeddings:
        pbar.update(0)  # 진행 표시줄 초기화
    
    # 동시 요청을 처리할 ThreadPoolExecutor 생성
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 텍스트 배치로 분할
        text_items = [(start_index + i, text) for i, text in enumerate(texts)]
        batches = [text_items[i:i+batch_size] for i in range(0, len(text_items), batch_size)]
        
        # 배치 처리 함수
        def process_batch(batch):
            batch_results = []
            for idx, text in batch:
                try:
                    embedding = get_embedding_with_retry(text, model_name, client)
                    batch_results.append((idx, embedding))
                except Exception as e:
                    print(f"\n임베딩 생성 오류 (인덱스 {idx}): {e}")
                    # 오류 시 0 벡터
                    batch_results.append((idx, [0.0] * 1536))
            return batch_results
        
        # 체크포인트 저장 함수
        def save_checkpoint(current_embeddings, max_index):
            with open(checkpoint_file, 'wb') as f:
                checkpoint_data = {
                    'embeddings': current_embeddings,
                    'last_index': max_index,
                    'timestamp': time.time()
                }
                pickle.dump(checkpoint_data, f)
            print(f"\n체크포인트 저장 완료 (인덱스: {max_index}, 임베딩: {len(current_embeddings)}개)")
        
        # 각 배치에 대한 작업 제출
        futures = []
        for batch in batches:
            future = executor.submit(process_batch, batch)
            futures.append(future)
        
        # 완료된 작업 결과 수집 및 정렬
        results_dict = {}  # 인덱스를 키로 사용하여 결과 저장
        checkpoint_counter = 0
        
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                
                # 결과 사전에 추가
                for idx, embedding in batch_results:
                    results_dict[idx] = embedding
                
                # 진행 상황 업데이트
                pbar.update(len(batch_results))
                
                # 체크포인트 저장 여부 확인
                checkpoint_counter += len(batch_results)
                if checkpoint_counter >= checkpoint_interval:
                    # 처리된 모든 결과를 인덱스 순으로 정렬하여 임베딩 목록 생성
                    sorted_indices = sorted(results_dict.keys())
                    sorted_embeddings = [results_dict[idx] for idx in sorted(results_dict.keys())]
                    
                    # 기존 임베딩과 새 임베딩 합치기
                    all_embeddings = existing_embeddings + sorted_embeddings
                    
                    # 체크포인트 저장
                    save_checkpoint(all_embeddings, sorted_indices[-1] if sorted_indices else last_processed_index)
                    
                    # 카운터 초기화
                    checkpoint_counter = 0
                    
                    # 메모리 관리를 위해 results_dict 비우기 (이미 체크포인트에 저장됨)
                    results_dict = {}
            except Exception as e:
                print(f"\n배치 처리 오류: {e}")
        
        pbar.close()
        
        # 최종 결과 수집 및 체크포인트 저장
        if results_dict:
            sorted_indices = sorted(results_dict.keys())
            sorted_embeddings = [results_dict[idx] for idx in sorted_indices]
            
            # 기존 임베딩과 새 임베딩 합치기
            all_embeddings = existing_embeddings + sorted_embeddings
            
            # 최종 체크포인트 저장
            final_index = sorted_indices[-1] if sorted_indices else last_processed_index
            save_checkpoint(all_embeddings, final_index)
            
            return all_embeddings
        else:
            # 모든 결과가 이미 체크포인트에 저장된 경우
            return existing_embeddings


# In[24]:


# OpenAI 클라이언트 설정
client = OpenAI(api_key=OPENAI_API_KEY)
model_name = "text-embedding-ada-002"


# In[25]:


# 데이터 로드
df = pd.read_csv(input_file)
print(f"데이터 로드 완료: {len(df)}개 항목")

# 샘플링 (필요한 경우)
sampled_df = sample_data(df, sample_size)
print(f"샘플링된 데이터: {len(sampled_df)}개 항목")


# In[ ]:


# # 병렬 처리를 사용한 임베딩 생성

#### 다 했음 !!! #######


# print("병렬 임베딩 생성을 시작합니다...")
# start_time = time.time()

# # 병렬 작업자 수 설정 (API 속도 제한 고려)
# max_workers = 20  # API 속도 제한에 따라 조정

# # 배치 크기 설정 (한 번에 요청할 텍스트 수)
# batch_size = 100  # API 제한에 따라 조정

# # 병렬 임베딩 생성 실행
# embeddings = create_embeddings_parallel_with_checkpoints(
#     sampled_df, 
#     '통합_텍스트', 
#     "text-embedding-ada-002", 
#     client,
#     batch_size=batch_size,
#     max_workers=max_workers
# )

# # 소요 시간 계산 (이어서 처리한 부분만)
# elapsed_time = time.time() - start_time
# print(f"임베딩 생성 완료: {len(embeddings)}개")
# print(f"총 소요 시간: {elapsed_time/60:.2f}분")

# # 최종 임베딩 저장
# embeddings_file = "product_embeddings.pkl"
# save_embeddings_to_file(embeddings, embeddings_file)


# In[ ]:





# In[39]:


# 체크포인트에서 임베딩 로드
def load_embeddings_from_checkpoint(checkpoint_dir="embeddings_checkpoints"):
    """체크포인트 파일에서 임베딩을 로드합니다."""
    checkpoint_file = os.path.join(checkpoint_dir, "embeddings_checkpoint.pkl")
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                embeddings = checkpoint_data.get('embeddings', [])
                last_index = checkpoint_data.get('last_index', -1)
                
                print(f"체크포인트에서 {len(embeddings)}개 임베딩을 로드했습니다.")
                print(f"마지막 처리된 인덱스: {last_index}")
                
                return embeddings, last_index
        except Exception as e:
            print(f"체크포인트 로드 중 오류 발생: {e}")
    
    print("체크포인트 파일을 찾을 수 없습니다.")
    return [], -1


# In[40]:


# 체크포인트에서 임베딩 로드
print("체크포인트에서 임베딩 로드 중...")
embeddings, last_index = load_embeddings_from_checkpoint()


# In[41]:


# 5. Qdrant 벡터 저장소 설정
collection_name = "product_recommendations"
    
# 6. 임베딩을 Qdrant에 저장
store_embeddings_in_qdrant(collection_name, sampled_df, embeddings)

print("임베딩 모델 및 벡터 저장소 구축이 완료되었습니다.")


# ## 임베딩 시각화

# In[42]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from tqdm import tqdm
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm


# In[43]:


# 1. 임베딩 데이터 로드
def load_embeddings(embeddings_file):
    """
    Load embeddings from saved file.
    """
    try:
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings loaded: {len(embeddings)} items")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

# 2. 유효한 임베딩 필터링 및 NumPy 배열로 변환
def prepare_embeddings(embeddings):
    """
    Convert embeddings to NumPy array and filter valid ones.
    """
    # 유효한 임베딩만 필터링
    valid_embeddings = [emb for emb in embeddings if emb is not None]
    if len(valid_embeddings) == 0:
        print("No valid embeddings found")
        return None
    
    # NumPy 배열로 변환
    return np.array(valid_embeddings)

# 3. 차원 축소 (PCA)
def reduce_dimensions_pca(embeddings_array, n_components=2):
    """
    Reduce dimensions using PCA.
    """
    if embeddings_array is None:
        return None, None
    
    # 차원 축소
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_array)
    
    # 설명된 분산 비율
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance) * 100:.2f}%")
    
    return reduced_embeddings, pca

# 4. 차원 축소 (t-SNE)
def reduce_dimensions_tsne(embeddings_array, n_components=2, perplexity=30, max_iter=1000):
    """
    Reduce dimensions using t-SNE.
    """
    if embeddings_array is None:
        return None
    
    print("Running t-SNE dimension reduction... (this may take a while)")
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, embeddings_array.shape[0] - 1),
        max_iter=max_iter,
        random_state=42,
        verbose=1
    )
    reduced_embeddings = tsne.fit_transform(embeddings_array)
    return reduced_embeddings

# 5. 카테고리 이름 영어로 변환
def translate_categories(df, category_column='카테고리명'):
    """
    Translate category names to English.
    """
    if category_column not in df.columns:
        return df
    
    # 카테고리 이름 매핑 (한글 -> 영어)
    category_map = {}
    
    # 기본적인 카테고리 매핑 정의
    # 실제 데이터에 맞게 이 부분을 확장하세요
    korean_to_english = {
        '패션잡화': 'Fashion Accessories',
        '여성가방': 'Women\'s Bags',
        '크로스백': 'Crossbody Bags',
        '여성의류': 'Women\'s Clothing',
        '남성의류': 'Men\'s Clothing',
        '아동의류': 'Children\'s Clothing',
        '신발': 'Shoes',
        '액세서리': 'Accessories',
        '주방용품': 'Kitchen Items',
        '생활용품': 'Household Goods',
        '가전제품': 'Electronics',
        '디지털': 'Digital Devices',
        '화장품': 'Cosmetics',
        '식품': 'Food',
        '가구': 'Furniture'
    }
    
    # 카테고리 이름 변환
    english_categories = []
    for category in df[category_column]:
        # 카테고리가 '>'로 나뉘어 있는 경우 (예: 패션잡화>여성가방>크로스백)
        if '>' in category:
            parts = category.split('>')
            translated_parts = []
            for part in parts:
                part = part.strip()
                if part in korean_to_english:
                    translated_parts.append(korean_to_english[part])
                else:
                    # 매핑되지 않은 카테고리는 원본 이름 사용
                    translated_parts.append(f"Category_{hash(part) % 1000}")
                    category_map[part] = f"Category_{hash(part) % 1000}"
            english_categories.append(' > '.join(translated_parts))
        else:
            if category in korean_to_english:
                english_categories.append(korean_to_english[category])
            else:
                # 매핑되지 않은 카테고리는 원본 이름 사용
                english_categories.append(f"Category_{hash(category) % 1000}")
                category_map[category] = f"Category_{hash(category) % 1000}"
    
    # 번역된 카테고리로 대체
    df['Category_English'] = english_categories
    
    # 매핑 결과 출력 (디버깅용)
    if category_map:
        print("\nUnmapped categories (using hash codes):")
        for k, v in category_map.items():
            print(f"  {k} -> {v}")
    
    return df

# 6. 시각화 (정적 PDF)
def visualize_embeddings_to_pdf(reduced_embeddings, df, category_column='Category_English', 
                             method_name='PCA', output_file='embedding_visualization.pdf'):
    """
    Visualize reduced embeddings to PDF file.
    """
    if reduced_embeddings is None or len(reduced_embeddings) == 0:
        print("No embeddings to visualize")
        return
    
    # PDF 파일 설정
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
    
    # 첫 번째 페이지: 모든 포인트를 단일 색상으로 표시
    plt.figure(figsize=(14, 10))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        alpha=0.5,
        s=20,
        c='blue'
    )
    plt.title(f'Product Embedding Visualization ({method_name})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # 두 번째 페이지: 카테고리별 색상 표시
    if category_column in df.columns:
        plt.figure(figsize=(14, 10))
        
        # 카테고리가 너무 많으면 상위 N개만 표시
        max_categories = 10
        categories = df[category_column].unique()
        
        if len(categories) > max_categories:
            # 가장 빈도가 높은 카테고리 선택
            top_categories = df[category_column].value_counts().head(max_categories).index.tolist()
            category_mask = df[category_column].isin(top_categories)
            
            # 나머지는 '기타'로 처리
            other_mask = ~category_mask
            if sum(other_mask) > 0:
                plt.scatter(
                    reduced_embeddings[other_mask, 0],
                    reduced_embeddings[other_mask, 1],
                    label='Other',
                    alpha=0.3,
                    s=15,
                    c='lightgray'
                )
            
            # 상위 카테고리만 처리
            categories = top_categories
        
        # 카테고리별 색상 지정
        colors = cm.tab10(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            mask = df[category_column] == category
            if sum(mask) > 0:
                plt.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    label=category,
                    alpha=0.7,
                    s=30,
                    c=[colors[i]]
                )
        
        plt.title(f'Product Embedding by Category ({method_name})', fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # 세 번째 페이지: 클러스터링 시각화
    try:
        from sklearn.cluster import KMeans
        
        # K-means 클러스터링
        n_clusters = min(8, len(reduced_embeddings) // 10 + 1)  # 데이터 크기에 따라 적절한 클러스터 수 조정
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(reduced_embeddings)
        
        plt.figure(figsize=(14, 10))
        
        # 클러스터별 색상 지정
        colors = cm.tab10(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = clusters == i
            if sum(mask) > 0:
                plt.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    label=f'Cluster {i+1}',
                    alpha=0.7,
                    s=30,
                    c=[colors[i]]
                )
        
        plt.title(f'Product Embedding Clusters ({method_name})', fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 클러스터 분석 텍스트 페이지
        fig = plt.figure(figsize=(10, 14))
        ax = plt.gca()
        ax.axis('off')
        ax.text(0.05, 0.95, "Cluster Analysis Results", fontsize=16, weight='bold')
        
        y_pos = 0.9
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_size = sum(cluster_mask)
            
            info_text = f"\n\nCluster {i+1} (Products: {cluster_size})\n"
            
            # 카테고리 분포
            if category_column in df.columns:
                category_counts = df.loc[cluster_mask, category_column].value_counts().head(3)
                info_text += f"Top categories: {', '.join([f'{cat} ({count})' for cat, count in category_counts.items()])}\n"
            
            # 가격 분포 (있는 경우)
            if '오너클랜판매가' in df.columns:
                price_mean = df.loc[cluster_mask, '오너클랜판매가'].mean()
                price_min = df.loc[cluster_mask, '오너클랜판매가'].min()
                price_max = df.loc[cluster_mask, '오너클랜판매가'].max()
                info_text += f"Price range: Avg {price_mean:.0f} KRW (Min {price_min:.0f} ~ Max {price_max:.0f} KRW)\n"
            
            # 대표 상품 (있는 경우)
            if '원본상품명' in df.columns:
                # 영어로 표시하기 위해 짧은 상품명 사용
                sample_products = df.loc[cluster_mask, '원본상품명'].sample(min(3, cluster_size)).tolist()
                sample_products = [f"Product {i+1}" for i, _ in enumerate(sample_products)]
                info_text += f"Example products: {', '.join(sample_products)}"
            
            ax.text(0.05, y_pos, info_text, fontsize=10)
            y_pos -= 0.1
        
        pdf.savefig(fig)
        plt.close()
    
    except Exception as e:
        print(f"Error during clustering visualization: {e}")
    
    # PDF 파일 닫기
    pdf.close()
    print(f"Visualization saved to {output_file}")


# In[48]:


# 메인 함수
# 파일 경로 설정
embeddings_file = "embeddings_checkpoints/embeddings_checkpoint.pkl"
processed_data_file = "narosu_db_final.csv"


# In[49]:


# 1. 임베딩 데이터 로드
raw_embeddings = load_embeddings(embeddings_file)
if raw_embeddings is None:
    print("Could not load embedding data")

# 2. 유효한 임베딩 준비 및 NumPy 배열로 변환
embeddings_array = prepare_embeddings(raw_embeddings)
if embeddings_array is None:
    print("No valid embeddings")


# In[50]:


# 3. 처리된 데이터 로드
try:
    df = pd.read_csv(processed_data_file)
    print(f"Product data loaded: {len(df)} items")
    
    # 데이터와 임베딩 길이 확인
    if len(df) != len(raw_embeddings):
        print(f"Warning: Data length ({len(df)}) does not match embedding length ({len(raw_embeddings)})")
        # 더 작은 크기로 조정
        min_length = min(len(df), len(embeddings_array))
        df = df.iloc[:min_length]
        embeddings_array = embeddings_array[:min_length]
        print(f"Adjusted data to {min_length} items")
    
    # 카테고리 이름 영어로 변환
    df = translate_categories(df, category_column='카테고리명')
    
except Exception as e:
    print(f"Error loading product data: {e}")


# In[51]:


# 4. 시각화 디렉토리 생성
viz_dir = "visualization"
os.makedirs(viz_dir, exist_ok=True)

# 5. PCA 차원 축소 및 시각화
reduced_embeddings_pca, _ = reduce_dimensions_pca(embeddings_array)
visualize_embeddings_to_pdf(
    reduced_embeddings_pca, 
    df, 
    category_column='Category_English',
    method_name='PCA',
    output_file=os.path.join(viz_dir, 'embedding_pca.pdf')
)


# In[131]:


# 6. t-SNE 차원 축소 및 시각화 (선택적)
do_tsne = input("Run t-SNE visualization? (y/n): ").lower() == 'y'
if do_tsne:
    try:
        reduced_embeddings_tsne = reduce_dimensions_tsne(
            embeddings_array,
            perplexity=min(30, len(embeddings_array) - 1),
            max_iter=1000
        )
        
        visualize_embeddings_to_pdf(
            reduced_embeddings_tsne, 
            df, 
            category_column='Category_English',
            method_name='t-SNE',
            output_file=os.path.join(viz_dir, 'embedding_tsne.pdf')
        )
    except Exception as e:
        print(f"Error during t-SNE visualization: {e}")

print(f"\nVisualization complete! Results saved to '{viz_dir}' directory")


# ## 대용량 데이터 임베딩 일괄 처리 및 저장

# In[53]:


get_ipython().system('pip install dotenv')


# In[54]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv
import time
import pickle
from datetime import datetime
import logging
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models


# In[56]:


def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='대용량 상품 데이터 임베딩 처리')
    parser.add_argument('--input', type=str, default='processed_data.csv', 
                        help='전처리된 상품 데이터 파일 경로')
    parser.add_argument('--original_input', type=str, default='data.xlsx',
                        help='원본 엑셀 데이터 파일 경로 (전처리 필요시)')
    parser.add_argument('--output', type=str, default='embeddings', 
                        help='임베딩 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=100, 
                        help='OpenAI API 배치 크기')
    parser.add_argument('--chunk_size', type=int, default=10000, 
                        help='데이터 청크 크기')
    parser.add_argument('--text_column', type=str, default='통합_텍스트', 
                        help='임베딩할 텍스트 컬럼')
    parser.add_argument('--model', type=str, default='text-embedding-ada-002', 
                        help='OpenAI 임베딩 모델')
    parser.add_argument('--qdrant_url', type=str, default=None, 
                        help='Qdrant 서버 URL (없으면 로컬 메모리 모드)')
    parser.add_argument('--collection_name', type=str, default='product_recommendations', 
                        help='Qdrant 컬렉션 이름')
    parser.add_argument('--preprocess', action='store_true',
                        help='원본 데이터를 먼저 전처리할지 여부')
    
    return parser.parse_args()


# In[57]:


import sys
sys.argv = [sys.argv[0]]  # 모든 Jupyter 관련 인수 제거

# 필요한 인수 직접 설정
sys.argv.extend(['--original_input', 'data.xlsx'])
sys.argv.extend(['--output', 'embeddings'])
sys.argv.extend(['--batch_size', '50'])
sys.argv.extend(['--preprocess'])

# 이후 parse_arguments() 호출
args = parse_arguments()


# In[58]:


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("secret")


# In[59]:


def setup_qdrant(collection_name, qdrant_url=None, vector_size=1536):
    """Qdrant 벡터 저장소를 설정합니다."""
    try:
        # Qdrant 클라이언트 생성
        if qdrant_url:
            client = QdrantClient(url=qdrant_url)
            logger.info(f"Qdrant 서버에 연결: {qdrant_url}")
        else:
            client = QdrantClient(":memory:")
            logger.info("로컬 메모리 모드로 Qdrant 실행")
        
        # 기존 컬렉션 확인
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        # 컬렉션이 없으면 생성
        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Qdrant 컬렉션 '{collection_name}'이(가) 생성되었습니다.")
        else:
            logger.info(f"Qdrant 컬렉션 '{collection_name}'이(가) 이미 존재합니다.")
        
        return client
    except Exception as e:
        logger.error(f"Qdrant 설정 중 오류 발생: {e}")
        return None


# In[60]:


def get_embedding_with_retry(client, text, model, max_retries=3, backoff_factor=2):
    """재시도 로직이 포함된 임베딩 함수"""
    if pd.isna(text) or text == "":
        return None
    
    text = text.replace("\n", " ")
    retries = 0
    
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            wait_time = backoff_factor ** retries
            logger.warning(f"임베딩 오류 (재시도 {retries+1}/{max_retries}): {e}")
            logger.warning(f"{wait_time}초 후 재시도...")
            time.sleep(wait_time)
            retries += 1
    
    logger.error(f"최대 재시도 횟수 초과. 임베딩 실패.")
    return None


# In[61]:


def process_chunk(df_chunk, args, openai_client, qdrant_client, chunk_idx):
    """데이터 청크를 처리하고 임베딩을 생성합니다."""
    start_time = time.time()
    logger.info(f"청크 {chunk_idx} 처리 시작: {len(df_chunk)} 개의 상품")
    
    # 임베딩 생성 (배치로 처리)
    embeddings = []
    errors = 0
    
    # 배치 처리
    for i in tqdm(range(0, len(df_chunk), args.batch_size), desc=f"청크 {chunk_idx} 임베딩"):
        batch_df = df_chunk.iloc[i:i+args.batch_size]
        batch_texts = batch_df[args.text_column].tolist()
        
        # 각 텍스트에 대한 임베딩 생성
        batch_embeddings = []
        for text in batch_texts:
            embedding = get_embedding_with_retry(openai_client, text, args.model)
            
            if embedding:
                batch_embeddings.append(embedding)
            else:
                # 오류 발생 시 0 벡터 사용
                batch_embeddings.append([0.0] * 1536)
                errors += 1
        
        embeddings.extend(batch_embeddings)
        
        # API 속도 제한 방지를 위한 대기
        time.sleep(0.5)
    
    # 진행 상황 및 통계 보고
    elapsed_time = time.time() - start_time
    logger.info(f"청크 {chunk_idx} 임베딩 생성 완료: {len(embeddings)} 개, 오류: {errors}개")
    logger.info(f"소요 시간: {elapsed_time:.2f}초 (평균: {elapsed_time/len(df_chunk):.2f}초/상품)")
    
    # 임베딩 저장 (파일로)
    chunk_output_file = os.path.join(args.output, f"embeddings_chunk_{chunk_idx}.pkl")
    with open(chunk_output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'product_ids': df_chunk['상품코드'].tolist() if '상품코드' in df_chunk.columns else list(range(len(df_chunk)))
        }, f)
    logger.info(f"청크 {chunk_idx} 임베딩 파일 저장 완료: {chunk_output_file}")
    
    # Qdrant에 저장
    if qdrant_client:
        points = []
        for i, embedding in enumerate(embeddings):
            if embedding:
                # 메타데이터 구성
                metadata = {
                    "상품코드": str(df_chunk.iloc[i].get("상품코드", "")),
                    "원본상품명": str(df_chunk.iloc[i].get("원본상품명", "")),
                    "카테고리명": str(df_chunk.iloc[i].get("카테고리명", "")),
                    "가격": int(df_chunk.iloc[i].get("오너클랜판매가", 0)),
                    "키워드": str(df_chunk.iloc[i].get("키워드", "")),
                    "이미지URL": str(df_chunk.iloc[i].get("이미지대", ""))
                }
                
                # 고유 ID 생성 (청크 인덱스와 로컬 인덱스 조합)
                unique_id = chunk_idx * args.chunk_size + i
                
                point = models.PointStruct(
                    id=unique_id,
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)
        
        # 배치 업로드
        if points:
            try:
                qdrant_client.upsert(
                    collection_name=args.collection_name,
                    points=points
                )
                logger.info(f"청크 {chunk_idx}의 {len(points)}개 임베딩을 Qdrant에 저장 완료")
            except Exception as e:
                logger.error(f"Qdrant 저장 오류 (청크 {chunk_idx}): {e}")
    
    return embeddings


# In[62]:


def preprocess_original_data(input_file, output_file):
    """원본 엑셀 데이터를 전처리하여 CSV로 저장합니다."""
    logger.info(f"원본 엑셀 파일 '{input_file}' 전처리 시작...")
    
    # 엑셀 파일 로드
    try:
        if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            try:
                df = pd.read_excel(input_file, engine='openpyxl')
            except Exception as e:
                logger.warning(f"openpyxl 엔진으로 로드 실패: {e}. xlrd 엔진 시도 중...")
                df = pd.read_excel(input_file, engine='xlrd')
        else:
            df = pd.read_csv(input_file, encoding='utf-8')
        
        logger.info(f"데이터 로드 완료: {len(df)}개의 상품")
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return None
    
    # 주요 필드 전처리
    text_columns = ['원본상품명', '카테고리명', '키워드', '정보고시 항목정보', '모델명', '제작/수입사']
    
    # 텍스트 전처리 함수
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        # 특수문자 제거 (단, 한글, 영문, 숫자, 일부 특수문자는 유지)
        import re
        text = re.sub(r'[^\wㄱ-ㅎㅏ-ㅣ가-힣\s\,\.\-\/]', ' ', text)
        
        # 여러 공백을 하나의 공백으로 대체
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    # 텍스트 필드 전처리
    for col in text_columns:
        if col in df.columns:
            logger.info(f"'{col}' 컬럼 전처리 중...")
            df[col] = df[col].apply(preprocess_text)
    
    # 결측치 처리
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
    
    # 상품 설명 텍스트 통합
    logger.info("통합 텍스트 생성 중...")
    df['통합_텍스트'] = (
        df['원본상품명'] + ' ' + 
        df['카테고리명'] + ' ' + 
        df['키워드'] + ' ' + 
        df['모델명'].astype(str) + ' ' + 
        df['정보고시 항목정보'].astype(str)
    )
    
    # 가격 정보 정규화
    if '오너클랜판매가' in df.columns:
        price_max = df['오너클랜판매가'].max()
        price_min = df['오너클랜판매가'].min()
        df['가격_정규화'] = (df['오너클랜판매가'] - price_min) / (price_max - price_min if price_max > price_min else 1)
    
    # 전처리된 데이터 저장
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"전처리된 데이터를 '{output_file}'에 저장 완료")
    
    return df


# In[63]:


# 인수 파싱
args = parse_arguments()

# 출력 디렉토리 생성
os.makedirs(args.output, exist_ok=True)


# In[64]:


# OpenAI 클라이언트 설정
openai_client = OpenAI(api_key="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")
logger.info(f"임베딩 모델: {args.model}")

# Qdrant 설정
qdrant_client = setup_qdrant(args.collection_name, args.qdrant_url)


# In[65]:


# 데이터 로드 또는 전처리
if args.preprocess:
    logger.info("원본 데이터 전처리 모드 활성화")
    df = preprocess_original_data(args.original_input, args.input)
    if df is None:
        logger.error("전처리 실패. 프로그램 종료.")

else:
    # 이미 전처리된 데이터 로드
    logger.info(f"전처리된 데이터 파일 '{args.input}' 로드 중...")
    try:
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
        else:
            # 확장자에 따라 적절한 로더 사용
            df = pd.read_excel(args.input)
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        
    logger.info(f"총 {len(df)}개의 상품 데이터 로드 완료")

# 데이터를 청크로 분할하여 처리
num_chunks = (len(df) + args.chunk_size - 1) // args.chunk_size
logger.info(f"데이터를 {num_chunks}개의 청크로 분할하여 처리합니다 (청크 크기: {args.chunk_size})")

# 전체 처리 시작 시간
total_start_time = time.time()

# 각 청크 처리
for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * args.chunk_size
    end_idx = min((chunk_idx + 1) * args.chunk_size, len(df))
    df_chunk = df.iloc[start_idx:end_idx]
    
    logger.info(f"청크 {chunk_idx+1}/{num_chunks} 처리 시작 (행 {start_idx}-{end_idx-1})")
    
    # 청크 처리
    process_chunk(df_chunk, args, openai_client, qdrant_client, chunk_idx)
    
# 전체 처리 완료 시간
total_elapsed_time = time.time() - total_start_time
logger.info(f"모든 청크 처리 완료! 총 소요 시간: {total_elapsed_time/60:.2f}분")
logger.info(f"임베딩 파일이 '{args.output}' 디렉토리에 저장되었습니다.")

# 처리 결과 요약
logger.info(f"처리 결과 요약:")
logger.info(f"- 총 상품 수: {len(df)}")
logger.info(f"- 총 청크 수: {num_chunks}")
logger.info(f"- 임베딩 모델: {args.model}")
logger.info(f"- 벡터 저장소: {'Qdrant (메모리 모드)' if args.qdrant_url is None else args.qdrant_url}")


# In[ ]:





# ## RAG 시스템

# In[140]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import time
from dotenv import load_dotenv
import logging
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, PointStruct

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class ProductRAGSystem:
    """
    상품 추천 및 검색을 위한 RAG (Retrieval Augmented Generation) 시스템
    """
    
    def __init__(self, 
                 openai_api_key=None,
                 qdrant_url=None, 
                 collection_name="product_recommendations",
                 embedding_model="text-embedding-ada-002",
                 llm_model="gpt-3.5-turbo",
                 vector_size=1536):
        """
        RAG 시스템 초기화
        
        Args:
            openai_api_key (str): OpenAI API 키
            qdrant_url (str): Qdrant 서버 URL (None이면 로컬 메모리 모드)
            collection_name (str): Qdrant 컬렉션 이름
            embedding_model (str): 임베딩에 사용할 OpenAI 모델
            llm_model (str): 텍스트 생성에 사용할 OpenAI 모델
            vector_size (int): 임베딩 벡터 크기
        """
        # OpenAI 클라이언트 설정
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Qdrant 클라이언트 설정
        if qdrant_url:
            self.qdrant_client = QdrantClient(url=qdrant_url)
            logger.info(f"Qdrant 서버에 연결: {qdrant_url}")
        else:
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("로컬 메모리 모드로 Qdrant 실행")
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        logger.info(f"RAG 시스템 초기화 완료 (collection: {collection_name}, model: {embedding_model})")
    
    def get_embedding(self, text):
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            list: 임베딩 벡터
        """
        if not text or pd.isna(text):
            return [0.0] * self.vector_size
        
        text = text.replace("\n", " ")
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            return [0.0] * self.vector_size
    
    def semantic_search(self, query_text, top_k=5, filters=None):
        """
        의미 기반 검색 수행
        
        Args:
            query_text (str): 검색 쿼리
            top_k (int): 반환할 최대 결과 수
            filters (dict): 검색 필터 (예: {"category": "의류"})
            
        Returns:
            list: 검색 결과 목록
        """
        # 쿼리 텍스트를 임베딩으로 변환
        query_vector = self.get_embedding(query_text)
        
        # 필터 구성
        search_filter = None
        if filters:
            filter_conditions = []
            
            # 카테고리 필터
            if "category" in filters and filters["category"]:
                filter_conditions.append(
                    FieldCondition(
                        key="카테고리명",
                        match={"text": filters["category"]}
                    )
                )
            
            # 가격 범위 필터
            if "price_min" in filters and "price_max" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="가격",
                        range=Range(
                            gte=filters["price_min"],
                            lte=filters["price_max"]
                        )
                    )
                )
            
            # 필터 조합
            if filter_conditions:
                search_filter = Filter(
                    must=filter_conditions
                )
        
        # 벡터 검색 수행
        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k
            )
            
            # 결과 형식 변환
            results = []
            for res in search_result:
                results.append({
                    "id": res.id,
                    "score": res.score,
                    "payload": res.payload
                })
            
            return results
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return []
    
    def hybrid_search(self, query_text, top_k=5, semantic_weight=0.7, filters=None):
        """
        하이브리드 검색 수행 (의미 검색 + 키워드 검색)
        
        Args:
            query_text (str): 검색 쿼리
            top_k (int): 반환할 최대 결과 수
            semantic_weight (float): 의미 검색의 가중치 (0~1 사이)
            filters (dict): 검색 필터
            
        Returns:
            list: 검색 결과 목록
        """
        # TODO: 키워드 검색 로직 추가 및 의미 검색과 결합
        # 현재는 단순히 의미 검색 결과만 반환
        return self.semantic_search(query_text, top_k, filters)
    
    def generate_product_recommendations(self, query_text, user_profile=None, top_k=5, temperature=0.7):
        """
        사용자 쿼리에 기반한 제품 추천 및 설명 생성
        
        Args:
            query_text (str): 사용자 쿼리
            user_profile (dict): 사용자 프로필 정보
            top_k (int): 추천할 상품 수
            temperature (float): 생성 모델의 temperature
            
        Returns:
            dict: 추천 상품 목록 및 설명
        """
        # 1. 검색 필터 구성
        filters = {}
        if user_profile:
            # 사용자 프로필 기반 필터 추가
            if "preferred_categories" in user_profile:
                filters["category"] = user_profile["preferred_categories"]
            
            if "price_range" in user_profile:
                filters["price_min"] = user_profile["price_range"][0]
                filters["price_max"] = user_profile["price_range"][1]
        
        # 2. 관련 상품 검색
        relevant_products = self.semantic_search(query_text, top_k=top_k, filters=filters)
        
        if not relevant_products:
            return {
                "recommendations": [],
                "explanation": "죄송합니다. 요청하신 조건에 맞는 상품을 찾을 수 없습니다."
            }
        
        # 3. 상품 정보 포맷팅
        product_info = []
        for i, product in enumerate(relevant_products):
            product_info.append(
                f"{i+1}. {product['payload']['원본상품명']} (카테고리: {product['payload']['카테고리명']}, 가격: {product['payload']['가격']}원)"
            )
        
        product_descriptions = "\n".join(product_info)
        
        # 4. LLM을 사용하여 추천 설명 생성
        prompt = f"""
                사용자 요청: "{query_text}"
                
                다음 상품 목록을 기반으로 사용자에게 적절한 추천 이유와 함께 상품을 추천해주세요.
                각 상품의 장점과 사용자 요청에 얼마나 부합하는지 설명해 주세요.
                
                상품 목록:
                {product_descriptions}
                
                다음 형식으로 응답해 주세요:
                1. [추천 이유와 함께 상품에 대한 상세 설명]
                2. [추천 이유와 함께 상품에 대한 상세 설명]
                ...
                
                마지막에 전체 추천에 대한 요약 설명을 추가해 주세요.
                """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "당신은 전문 쇼핑 어드바이저입니다. 사용자의 요구에 맞는 상품을 추천하고 이유를 설명해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            explanation = response.choices[0].message.content
            
            # 5. 결과 반환
            return {
                "recommendations": relevant_products,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {e}")
            
            # 오류 발생 시 기본 설명 제공
            basic_explanation = f"요청하신 '{query_text}'에 관련된 상품을 찾았습니다. 다음 상품들을 확인해보세요:\n\n" + product_descriptions
            
            return {
                "recommendations": relevant_products,
                "explanation": basic_explanation
            }
    
    def personalized_search(self, query_text, user_id, top_k=5):
        """
        사용자 맞춤형 검색 수행
        
        Args:
            query_text (str): 검색 쿼리
            user_id (str): 사용자 ID
            top_k (int): 반환할 최대 결과 수
            
        Returns:
            list: 검색 결과 목록
        """
        # TODO: 사용자 프로필 및 행동 데이터 로드
        user_profile = self._load_user_profile(user_id)
        
        # 사용자 맞춤형 추천 생성
        return self.generate_product_recommendations(
            query_text=query_text,
            user_profile=user_profile,
            top_k=top_k
        )
    
    def _load_user_profile(self, user_id):
        """
        사용자 프로필 로드 (예시)
        
        실제 구현에서는 데이터베이스나 사용자 프로필 저장소에서 로드
        
        Args:
            user_id (str): 사용자 ID
            
        Returns:
            dict: 사용자 프로필 정보
        """
        # 예시 - 실제 구현에서는 DB에서 로드
        example_profiles = {
            "user1": {
                "preferred_categories": ["패션잡화", "여성가방"],
                "price_range": [10000, 100000],
                "interests": ["캐주얼", "데일리"]
            },
            "user2": {
                "preferred_categories": ["전자제품", "디지털"],
                "price_range": [50000, 500000],
                "interests": ["최신기술", "가성비"]
            }
        }
        
        return example_profiles.get(user_id, {})
    
    def add_product_to_index(self, product_data):
        """
        상품 데이터를 벡터 인덱스에 추가
        
        Args:
            product_data (dict): 상품 데이터
            
        Returns:
            bool: 성공 여부
        """
        # 1. 텍스트 필드 통합
        if not product_data.get("통합_텍스트"):
            product_data["통합_텍스트"] = (
                str(product_data.get("원본상품명", "")) + " " +
                str(product_data.get("카테고리명", "")) + " " +
                str(product_data.get("키워드", "")) + " " +
                str(product_data.get("모델명", "")) + " " +
                str(product_data.get("정보고시 항목정보", ""))
            )
        
        # 2. 임베딩 생성
        embedding = self.get_embedding(product_data["통합_텍스트"])
        
        # 3. Qdrant에 저장
        try:
            # 고유 ID 생성 (실제 구현에서는 고유 ID 로직 필요)
            product_id = product_data.get("상품코드", str(hash(product_data["원본상품명"])))
            
            # 메타데이터 구성
            metadata = {
                "상품코드": str(product_data.get("상품코드", "")),
                "원본상품명": str(product_data.get("원본상품명", "")),
                "카테고리명": str(product_data.get("카테고리명", "")),
                "가격": int(product_data.get("오너클랜판매가", 0)),
                "키워드": str(product_data.get("키워드", "")),
                "이미지URL": str(product_data.get("이미지대", ""))
            }
            
            point = PointStruct(
                id=product_id,
                vector=embedding,
                payload=metadata
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"상품 '{product_data['원본상품명']}' 인덱스 추가 완료")
            return True
        except Exception as e:
            logger.error(f"상품 인덱스 추가 오류: {e}")
            return False
    
    def chatbot_response(self, user_message, chat_history=None, user_id=None):
        """
        사용자 메시지에 대한 챗봇 응답 생성
        
        Args:
            user_message (str): 사용자 메시지
            chat_history (list): 이전 대화 기록
            user_id (str): 사용자 ID
            
        Returns:
            str: 챗봇 응답
        """
        # 1. 사용자 메시지 의도 분석
        is_product_query = self._is_product_search_query(user_message)
        
        if is_product_query:
            # 2. 제품 검색 의도가 있으면 추천 생성
            if user_id:
                # 사용자 맞춤형 추천
                recommendation_result = self.personalized_search(
                    query_text=user_message,
                    user_id=user_id,
                    top_k=3
                )
            else:
                # 일반 추천
                recommendation_result = self.generate_product_recommendations(
                    query_text=user_message,
                    top_k=3
                )
            
            return recommendation_result["explanation"]
        else:
            # 3. 일반 대화 응답 생성
            return self._generate_conversation_response(user_message, chat_history)
    
    def _is_product_search_query(self, query):
        """
        사용자 쿼리가 제품 검색 의도인지 판단
        
        Args:
            query (str): 사용자 쿼리
            
        Returns:
            bool: 제품 검색 의도 여부
        """
        # 제품 검색 관련 키워드 확인
        search_keywords = ["찾아", "추천", "상품", "제품", "물건", "구매", "살", "쇼핑", "가격",
                         "브랜드", "어디", "구할", "파는", "보여", "어떤", "검색"]
        
        # 간단한 휴리스틱: 키워드 포함 여부 확인
        for keyword in search_keywords:
            if keyword in query:
                return True
        
        # 더 정확한 구현을 위해 의도 분류 모델 사용 가능
        return False
    
    def _generate_conversation_response(self, user_message, chat_history=None):
        """
        일반 대화에 대한 응답 생성
        
        Args:
            user_message (str): 사용자 메시지
            chat_history (list): 이전 대화 기록
            
        Returns:
            str: 챗봇 응답
        """
        try:
            # 시스템 프롬프트 설정
            system_prompt = """
                당신은 친절한 쇼핑 어시스턴트입니다. 사용자의 질문에 대해 도움을 제공하세요.
                사용자가 상품에 대해 물어보면 상세한 추천을 제공하고, 일반적인 질문에는 간결하고 유용한 정보를 제공하세요.
                """
            
            # 대화 기록 포맷팅
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if chat_history:
                for msg in chat_history:
                    messages.append(msg)
            
            messages.append({"role": "user", "content": user_message})
            
            # 응답 생성
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"대화 응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다. 다시 시도해주세요."


# In[141]:


# RAG 시스템 사용 예제
def rag_system_demo():
    """
    RAG 시스템 데모
    """
    # API 키 설정 
    openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
    
    # RAG 시스템 초기화
    rag = ProductRAGSystem(
        openai_api_key=openai_api_key,
        qdrant_url=None,  # 로컬 메모리 모드
        collection_name="product_recommendations",
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo"
    )
    
    # 간단한 데모 데이터 추가
    demo_products = [
        {
            "상품코드": "P001",
            "원본상품명": "프리미엄 가죽 크로스백",
            "카테고리명": "패션잡화>여성가방>크로스백",
            "오너클랜판매가": 89000,
            "키워드": "가죽,크로스백,여성,고급,선물",
            "정보고시 항목정보": "천연가죽 소재, 생산국: 이탈리아",
            "이미지대": "http://example.com/image1.jpg"
        },
        {
            "상품코드": "P002",
            "원본상품명": "스마트 워치 갤럭시 핏",
            "카테고리명": "디지털>웨어러블>스마트워치",
            "오너클랜판매가": 129000,
            "키워드": "스마트워치,웨어러블,갤럭시,피트니스,건강",
            "정보고시 항목정보": "배터리 용량: 247mAh, 방수등급: 5ATM",
            "이미지대": "http://example.com/image2.jpg"
        },
        {
            "상품코드": "P003",
            "원본상품명": "울트라 HD 게이밍 모니터 27인치",
            "카테고리명": "디지털>PC주변기기>모니터",
            "오너클랜판매가": 349000,
            "키워드": "게이밍,모니터,울트라HD,144Hz,응답속도",
            "정보고시 항목정보": "해상도: 3840x2160, 패널: IPS, 주사율: 144Hz",
            "이미지대": "http://example.com/image3.jpg"
        },
        {
            "상품코드": "P004",
            "원본상품명": "프리미엄 캐시미어 니트 가디건",
            "카테고리명": "여성의류>가디건",
            "오너클랜판매가": 159000,
            "키워드": "캐시미어,니트,가디건,여성,고급,보온",
            "정보고시 항목정보": "소재: 캐시미어 100%, 원산지: 몽골",
            "이미지대": "http://example.com/image4.jpg"
        },
        {
            "상품코드": "P005",
            "원본상품명": "초경량 등산 배낭 30L",
            "카테고리명": "스포츠>등산>배낭",
            "오너클랜판매가": 78000,
            "키워드": "등산,배낭,경량,백팩,아웃도어,여행",
            "정보고시 항목정보": "용량: 30L, 무게: 850g, 생활방수",
            "이미지대": "http://example.com/image5.jpg"
        }
    ]
    
    # 데모 데이터 인덱싱
    print("데모 상품 인덱싱 중...")
    for product in demo_products:
        rag.add_product_to_index(product)
    
    print("\n=== RAG 시스템 데모 ===")
    print("(종료하려면 'exit' 입력)")
    
    chat_history = []
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n질문이나 검색어를 입력하세요: ")
        
        if user_input.lower() == 'exit':
            break
        
        # 응답 생성
        response = rag.chatbot_response(user_input, chat_history)
        
        # 대화 기록 업데이트
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        
        # 응답 출력
        print("\n🤖 응답:")
        print(response)
        
        # 대화 기록이 너무 길어지면 잘라내기
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]


# 데모 실행 (실행 시 주석 해제)
# if __name__ == "__main__":
#     rag_system_demo()


# RAG 시스템 API 예제 (FastAPI 사용)
def create_rag_api():
    """
    RAG 시스템 API 생성 (FastAPI 사용)
    """
    from fastapi import FastAPI, HTTPException, Query, Depends
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    
    app = FastAPI(title="상품 RAG 시스템 API")
    
    # API 키 설정
    openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
    
    # RAG 시스템 초기화
    rag = ProductRAGSystem(
        openai_api_key=openai_api_key,
        qdrant_url=os.getenv("QDRANT_URL"),  # 환경 변수에서 URL 가져오기
        collection_name="product_recommendations",
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo"
    )
    
    # 모델 정의
    class SearchRequest(BaseModel):
        query: str
        filters: Optional[Dict[str, Any]] = None
        top_k: int = 5
    
    class ChatRequest(BaseModel):
        message: str
        user_id: Optional[str] = None
        chat_history: Optional[List[Dict[str, str]]] = None
    
    class ProductData(BaseModel):
        상품코드: str
        원본상품명: str
        카테고리명: str
        오너클랜판매가: int
        키워드: Optional[str] = ""
        정보고시_항목정보: Optional[str] = ""
        이미지대: Optional[str] = ""
    
    # 엔드포인트 정의
    @app.get("/")
    def read_root():
        return {"message": "상품 RAG 시스템 API"}
    
    @app.post("/search")
    def search_products(request: SearchRequest):
        results = rag.semantic_search(
            query_text=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        return {"results": results}
    
    @app.post("/recommend")
    def recommend_products(request: SearchRequest):
        recommendations = rag.generate_product_recommendations(
            query_text=request.query,
            top_k=request.top_k
        )
        return recommendations
    
    @app.post("/chat")
    def chat(request: ChatRequest):
        response = rag.chatbot_response(
            user_message=request.message,
            chat_history=request.chat_history,
            user_id=request.user_id
        )
        return {"response": response}
    
    @app.post("/add-product")
    def add_product(product: ProductData):
        # 모델을 딕셔너리로 변환
        product_dict = product.dict()
        # 필드명 일치시키기
        product_dict["정보고시 항목정보"] = product_dict.pop("정보고시_항목정보")
        
        success = rag.add_product_to_index(product_dict)
        if success:
            return {"status": "success", "message": "상품이 성공적으로 추가되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="상품 추가 중 오류가 발생했습니다.")
    
    return app


# In[ ]:


# FastAPI 애플리케이션 생성
app = create_rag_api()

# FastAPI 실행 방법 (터미널에서):
uvicorn rag_implementation:app --reload


# In[ ]:





# In[ ]:




