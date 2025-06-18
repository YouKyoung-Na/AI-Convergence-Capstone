import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('SSL_CERT_DIR', None)

import gradio as gr
import pandas as pd
import numpy as np
import os
import re
import json
import datetime
from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pickle
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 임베딩 모델 설정
def get_embeddings():
    """OpenAI 임베딩 모델을 설정합니다."""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key="secret"  # 실제 API 키로 변경해주세요
        )
        return embeddings
    except Exception as e:
        logger.error(f"임베딩 모델 초기화 중 오류 발생: {e}")
        return None

# Qdrant 클라이언트 설정
def setup_qdrant_client(collection_name="product_recommendations"):
    """Qdrant 클라이언트를 설정하고 반환합니다."""
    try:
        # 로컬 Qdrant 인스턴스에 연결
        client = QdrantClient(":memory:")  # 메모리 모드 (실제 배포 시 서버 URL 사용)
        
        # 컬렉션 존재 여부 확인
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        # 컬렉션이 없으면 생성
        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # text-embedding-ada-002의 차원
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Qdrant 컬렉션 '{collection_name}'이(가) 생성되었습니다.")
        
        return client
    except Exception as e:
        logger.error(f"Qdrant 설정 중 오류 발생: {e}")
        return None

# 상품 데이터 로드
def load_product_data(file_path="narosu_db_final.csv", excel_path="narosu_db_final.xlsx"):
    """전처리된 상품 데이터를 로드합니다."""
    try:
        # CSV 파일 시도
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"CSV 상품 데이터 로드 완료: {len(df)}개의 상품")
            return df
        
        # CSV 파일이 없으면 Excel 파일 시도
        elif os.path.exists(excel_path):
            logger.info(f"CSV 파일 없음, Excel 파일 '{excel_path}'로드 시도 중...")
            df = pd.read_excel(excel_path)
            
            # Excel에서 로드한 데이터를 CSV로 저장 (향후 사용을 위해)
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Excel 상품 데이터 로드 완료: {len(df)}개의 상품")
            return df
        
        else:
            logger.error(f"상품 데이터 파일을 찾을 수 없음: CSV('{file_path}') 또는 Excel('{excel_path}')")
            return None
    except Exception as e:
        logger.error(f"상품 데이터 로드 실패: {e}")
        return None

# 임베딩 데이터 로드
# 임베딩 데이터 로드
def load_embeddings(embeddings_dir="embeddings_checkpoints"):
    """저장된 임베딩 데이터를 로드합니다."""
    all_embeddings = []
    all_product_ids = []
    
    try:
        # embeddings_checkpoint.pkl 파일 확인
        checkpoint_file = os.path.join(embeddings_dir, "embeddings_checkpoint.pkl")
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
                # checkpoint 구조 확인 및 임베딩 추출
                if isinstance(checkpoint_data, dict) and 'embeddings' in checkpoint_data:
                    all_embeddings = checkpoint_data['embeddings']
                    logger.info(f"체크포인트에서 {len(all_embeddings)}개 임베딩 로드됨")
                    
                    # 상품 ID 생성 또는 추출
                    if 'last_index' in checkpoint_data:
                        last_index = checkpoint_data['last_index']
                        all_product_ids = [str(i) for i in range(last_index + 1)]
                        logger.info(f"상품 ID 데이터 생성 완료: {len(all_product_ids)}개")
                else:
                    # 구조가 다른 경우 직접 사용
                    all_embeddings = checkpoint_data
                    all_product_ids = [str(i) for i in range(len(all_embeddings))]
                    logger.info(f"임베딩 데이터 로드 완료: {len(all_embeddings)}개 (리스트 형식)")
                
        # 기존 방식도 시도 (청크 파일 확인)
        elif os.path.exists(embeddings_dir):
            chunk_files = [f for f in os.listdir(embeddings_dir) if f.startswith("embeddings_chunk_") and f.endswith(".pkl")]
            
            if chunk_files:
                for chunk_file in chunk_files:
                    file_path = os.path.join(embeddings_dir, chunk_file)
                    with open(file_path, 'rb') as f:
                        chunk_data = pickle.load(f)
                        if isinstance(chunk_data, dict) and 'embeddings' in chunk_data and 'product_ids' in chunk_data:
                            all_embeddings.extend(chunk_data['embeddings'])
                            all_product_ids.extend(chunk_data['product_ids'])
                
                logger.info(f"청크 파일에서 임베딩 데이터 로드 완료: {len(all_embeddings)}개")
            else:
                logger.warning(f"임베딩 디렉토리 '{embeddings_dir}'에 임베딩 파일이 없습니다.")
        else:
            logger.warning(f"임베딩 디렉토리 '{embeddings_dir}'와 체크포인트 파일을 찾을 수 없습니다.")
        
        return all_embeddings, all_product_ids
    except Exception as e:
        logger.error(f"임베딩 데이터 로드 실패: {e}")
        return [], []

# 벡터 저장소 초기화
def init_vector_store(client, collection_name, product_data, embeddings_data, product_ids, embeddings_model):
    """벡터 저장소를 초기화합니다."""
    try:
        # Qdrant 벡터 저장소 초기화
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings_model
        )
        
        # 벡터 저장소에 데이터가 없으면 추가
        if client.count(collection_name=collection_name).count == 0 and embeddings_data:
            logger.info("벡터 저장소에 데이터 추가 중...")
            
            # 임베딩과 상품 데이터 매핑
            documents = []
            points = []
            
            for i, (emb, product_id) in enumerate(zip(embeddings_data, product_ids)):
                # 해당 상품 ID의 데이터 찾기
                product_row = product_data[product_data['상품코드'] == product_id] if '상품코드' in product_data.columns else None
                
                if product_row is not None and not product_row.empty:
                    product = product_row.iloc[0]
                    
                    # 메타데이터 구성
                    metadata = {
                        "상품코드": str(product.get("상품코드", "")),
                        "원본상품명": str(product.get("원본상품명", "")),
                        "카테고리명": str(product.get("카테고리명", "")),
                        "가격": int(product.get("오너클랜판매가", 0)),
                        "키워드": str(product.get("키워드", "")),
                        "이미지URL": str(product.get("이미지대", ""))
                    }
                    
                    # Document 객체 생성
                    doc = Document(
                        page_content=str(product.get("통합_텍스트", "")),
                        metadata=metadata
                    )
                    documents.append(doc)
                    
                    # Qdrant 포인트 생성
                    point = models.PointStruct(
                        id=i,
                        vector=emb,
                        payload=metadata
                    )
                    points.append(point)
            
            # 배치 업로드 (대량 데이터의 경우 청크로 나누어 업로드)
            chunk_size = 1000  # 한 번에 업로드할 포인트 수
            for i in range(0, len(points), chunk_size):
                chunk = points[i:i + chunk_size]
                client.upsert(
                    collection_name=collection_name,
                    points=chunk
                )
                logger.info(f"청크 {i//chunk_size + 1}: {len(chunk)}개의 포인트 추가 완료")
            
            logger.info(f"총 {len(points)}개의 포인트를 Qdrant에 추가했습니다.")
        
        return vector_store
    except Exception as e:
        logger.error(f"벡터 저장소 초기화 중 오류 발생: {e}")
        return None

# LLaMA 모델 초기화
def init_llm():
    """LLaMA 3.1 모델을 초기화합니다."""
    try:
        # Ollama를 통해 LLaMA 3.1 모델에 접근
        # Ollama가 실행 중이어야 함: ollama run llama3.1
        llm = Ollama(model="llama3.1")
        return llm
    except Exception as e:
        logger.error(f"LLM 초기화 중 오류 발생: {e}")
        return None

# RAG 검색 파이프라인 초기화
def init_rag_pipeline(vector_store, llm):
    """RAG 검색 파이프라인을 초기화합니다."""
    if vector_store is None or llm is None:
        logger.error("벡터 저장소 또는 LLM이 초기화되지 않았습니다.")
        return None
    
    try:
        # 프롬프트 템플릿 설정 (개선된 프롬프트)
        prompt_template = """
        당신은 상품 추천 및 검색 전문가입니다. 사용자의 질문에 친절하고 명확하게 답변해주세요.
        
        # 지침
        - 질문에 관련된 상품을 직접적으로 추천해주세요.
        - 사용자의 의도와 선호도를 고려하세요.
        - 가격, 품질, 특징 등을 종합적으로 분석해 추천해주세요.
        - 추천 이유를 간결하게 설명해주세요.
        - 상품에 대한 정보가 부족하면 솔직하게 말해주세요.
        
        # 사용자 질문
        {question}
        
        # 관련 상품 정보
        {context}
        
        # 답변
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context"]
        )
        
        # RetrievalQA 체인 생성 (검색 개수 증가)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # 상위 8개 결과 검색 (더 많은 후보 검색)
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        logger.error(f"RAG 파이프라인 초기화 중 오류 발생: {e}")
        return None

# 채팅 기록 저장 및 응답 생성
class ChatHistory:
    def __init__(self):
        self.messages = []
        self.system_prompt = """
        당신은 상품 추천 및 검색 전문가입니다. 사용자의 질문을 이해하고 관련 상품을 추천해주세요.
        상품에 관한 질문이면 검색 결과를 바탕으로 정확하게 답변해주세요.
        검색 결과가 없거나 충분하지 않다면, 그 사실을 솔직하게 알려주세요.
        일반적인 대화에도 친절하게 응답해주세요.
        사용자의 이전 대화 맥락을 고려하여 연속적인 대화 흐름을 유지하세요.
        """
        self.search_history = []  # 검색 기록 저장
        
    def add_message(self, role, content):
        """메시지를 기록에 추가합니다."""
        self.messages.append({"role": role, "content": content})
        
    def add_search_result(self, query, results):
        """검색 기록을 저장합니다."""
        self.search_history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "num_results": len(results),
            "top_categories": self._extract_top_categories(results)
        })
    
    def _extract_top_categories(self, results):
        """검색 결과에서 상위 카테고리를 추출합니다."""
        categories = {}
        for doc in results:
            category = doc.metadata.get("카테고리명", "").split(" > ")[0]
            if category:
                categories[category] = categories.get(category, 0) + 1
        
        # 상위 3개 카테고리 반환
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        return [cat for cat, count in top_categories]
    
    def get_messages(self):
        """모든 메시지를 반환합니다."""
        return self.messages
    
    def get_display_messages(self):
        """Gradio 채팅 인터페이스용 메시지 포맷을 반환합니다."""
        return [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) 
                for msg in self.messages]
    
    def format_prompt(self, query):
        """LLM에 보낼 프롬프트를 구성합니다."""
        prompt = self.system_prompt + "\n\n"
        
        # 이전 대화 컨텍스트 추가 (최근 5개 메시지만)
        context_messages = self.messages[-10:]
        for msg in context_messages:
            if msg["role"] == "user":
                prompt += f"사용자: {msg['content']}\n"
            else:
                prompt += f"시스템: {msg['content']}\n"
        
        # 현재 쿼리 추가
        prompt += f"사용자: {query}\n시스템: "
        
        return prompt
    
    def get_search_analytics(self):
        """검색 분석 데이터를 반환합니다."""
        if not self.search_history:
            return "검색 기록이 없습니다."
            
        # 카테고리 통계
        all_categories = []
        for search in self.search_history:
            all_categories.extend(search["top_categories"])
        
        category_counts = {}
        for category in all_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
            
        # 검색어 분석
        search_queries = [search["query"] for search in self.search_history]
        
        return {
            "total_searches": len(self.search_history),
            "top_categories": sorted(category_counts.items(), key=lambda x: x[1], reverse=True),
            "search_queries": search_queries
        }

# 검색 결과 포맷팅 함수
def format_search_results(result):
    """검색 결과를 사용자 친화적인 형식으로 변환합니다."""
    answer = result['result']
    source_docs = result.get('source_documents', [])
    
    # 검색된 상품 정보 추출
    products_info = []
    for i, doc in enumerate(source_docs, 1):
        metadata = doc.metadata
        product_name = metadata.get('원본상품명', '상품명 없음')
        category = metadata.get('카테고리명', '카테고리 없음')
        price = metadata.get('가격', 0)
        product_id = metadata.get('상품코드', '')
        keywords = metadata.get('키워드', '')
        image_url = metadata.get('이미지URL', '')
        
        # 이미지가 있는 경우 HTML 이미지 태그 추가
        image_html = ""
        if image_url:
            image_html = f'\n<img src="{image_url}" alt="{product_name}" style="width:200px;height:200px;object-fit:cover;border-radius:8px;margin:10px 0;">\n'
        
        # 개선된 상품 정보 포맷 (이미지 포함)
        product_info = f"【{i}】 **{product_name}**{image_html}"
        product_info += f"📂 **카테고리**: {category}\n"
        product_info += f"💰 **가격**: {price:,}원\n"
        product_info += f"🏷️ **키워드**: {keywords}\n"
        product_info += f"🆔 **상품코드**: {product_id}\n"
        
        products_info.append(product_info)
    
    # 결과 포맷팅
    if products_info:
        formatted_result = f"{answer}\n\n---\n\n📚 **관련 상품 정보**:\n\n" + "\n\n---\n\n".join(products_info)
    else:
        formatted_result = answer
    
    return formatted_result


# 가격대별 상품 분포 시각화 함수
def create_price_distribution_chart(search_results):
    """검색 결과의 가격 분포를 시각화합니다."""
    try:
        # 가격 데이터 추출
        prices = [doc.metadata.get('가격', 0) for doc in search_results]
        if not prices:
            return None
            
        # 가격대 구간 설정
        price_ranges = [0, 10000, 30000, 50000, 100000, 200000, float('inf')]
        labels = ['1만원 이하', '1~3만원', '3~5만원', '5~10만원', '10~20만원', '20만원 이상']
        
        # 가격대별 상품 개수 계산
        counts = [0] * len(labels)
        for price in prices:
            for i, (lower, upper) in enumerate(zip(price_ranges[:-1], price_ranges[1:])):
                if lower <= price < upper:
                    counts[i] += 1
                    break
        
        # 차트 생성
        plt.figure(figsize=(8, 5))
        plt.bar(labels, counts, color='skyblue')
        plt.xlabel('가격대')
        plt.ylabel('상품 수')
        plt.title('검색 결과 가격 분포')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 이미지 버퍼로 저장
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # 이미지를 base64로 인코딩
        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"<img src='data:image/png;base64,{img_str}' alt='가격 분포 차트'>"
    except Exception as e:
        logger.error(f"차트 생성 오류: {e}")
        return None

# 카테고리 분포 시각화 함수
def create_category_distribution_chart(search_results):
    """검색 결과의 카테고리 분포를 시각화합니다."""
    try:
        # 카테고리 데이터 추출
        categories = {}
        for doc in search_results:
            category = doc.metadata.get('카테고리명', '').split(' > ')[0]
            if category:
                categories[category] = categories.get(category, 0) + 1
        
        if not categories:
            return None
            
        # 상위 5개 카테고리만 선택
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 차트 생성
        plt.figure(figsize=(8, 5))
        plt.bar([cat for cat, count in top_categories], [count for cat, count in top_categories], color='lightgreen')
        plt.xlabel('카테고리')
        plt.ylabel('상품 수')
        plt.title('검색 결과 카테고리 분포')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 이미지 버퍼로 저장
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # 이미지를 base64로 인코딩
        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"<img src='data:image/png;base64,{img_str}' alt='카테고리 분포 차트'>"
    except Exception as e:
        logger.error(f"차트 생성 오류: {e}")
        return None

# 데모 모드 여부 확인
def check_demo_mode(file_path="narosu_db.csv", embeddings_dir="embeddings"):
    """데모 모드 여부를 확인합니다. 실제 데이터가 없으면 데모 모드로 실행됩니다."""
    data_exists = os.path.exists(file_path)
    embeddings_exist = os.path.exists(embeddings_dir) and os.listdir(embeddings_dir) if os.path.exists(embeddings_dir) else False
    
    if not data_exists or not embeddings_exist:
        logger.warning("실제 데이터 또는 임베딩을 찾을 수 없습니다. 데모 모드로 실행됩니다.")
        return True
    
    return False


# 데모 모드 여부 확인
def check_demo_mode(file_path="narosu_db_final.csv", embeddings_dir="embeddings_checkpoints"):
    """데모 모드 여부를 확인합니다. 실제 데이터가 없으면 데모 모드로 실행됩니다."""
    data_exists = os.path.exists(file_path)
    embeddings_exist = False
    
    # 임베딩 파일 확인
    if os.path.exists(embeddings_dir):
        # embeddings_checkpoint.pkl 파일 확인
        checkpoint_file = os.path.join(embeddings_dir, "embeddings_checkpoint.pkl")
        if os.path.exists(checkpoint_file):
            embeddings_exist = True
    
    if not data_exists:
        logger.warning(f"데이터 파일 '{file_path}'을 찾을 수 없습니다.")
    if not embeddings_exist:
        logger.warning(f"임베딩 파일을 '{embeddings_dir}'에서 찾을 수 없습니다.")
    
    if not data_exists or not embeddings_exist:
        logger.warning("실제 데이터 또는 임베딩을 찾을 수 없습니다. 데모 모드로 실행됩니다.")
        return True
    
    return False


# 사용자 쿼리 전처리
def preprocess_query(query):
    """사용자 쿼리를 전처리합니다."""
    # 불용어 제거
    stopwords = ["좀", "그냥", "일단", "그리고", "그래서", "그럼", "요", "혹시"]
    query_words = query.split()
    filtered_words = [word for word in query_words if word not in stopwords]
    
    # 특수 문자 제거
    clean_query = re.sub(r'[^\w\s가-힣]', ' ', ' '.join(filtered_words))
    
    # 연속된 공백 제거
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()
    
    return clean_query

# 검색 쿼리 확장
def expand_search_query(query):
    """검색 쿼리를 확장합니다."""
    expanded_queries = [query]
    
    # 가격 관련 쿼리 확장
    price_patterns = {
        r'(\d+)만원\s*(이하|미만|이하의)': lambda m: f"{m.group(1)}만원 이하",
        r'(\d+)만원\s*(이상|초과|이상의)': lambda m: f"{m.group(1)}만원 이상",
        r'(\d+)만원대': lambda m: f"{m.group(1)}만원대"
    }
    
    for pattern, replacement_func in price_patterns.items():
        match = re.search(pattern, query)
        if match:
            expanded_query = replacement_func(match)
            if expanded_query not in expanded_queries:
                expanded_queries.append(expanded_query)
    
    # 계절 관련 쿼리 확장
    season_map = {
        "봄": ["봄", "스프링", "3월", "4월", "5월"],
        "여름": ["여름", "서머", "6월", "7월", "8월"],
        "가을": ["가을", "오토닝", "9월", "10월", "11월"],
        "겨울": ["겨울", "윈터", "12월", "1월", "2월"]
    }
    
    for season, synonyms in season_map.items():
        if season in query:
            for synonym in synonyms:
                if synonym != season and synonym not in query:
                    new_query = query.replace(season, synonym)
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
    
    # 상품 유형 확장
    product_map = {
        "자켓": ["자켓", "코트", "아우터"],
        "가방": ["가방", "백", "클러치", "토트백", "숄더백"],
        "신발": ["신발", "슈즈", "운동화", "스니커즈"]
    }
    
    for product, synonyms in product_map.items():
        if product in query:
            for synonym in synonyms:
                if synonym != product and synonym not in query:
                    new_query = query.replace(product, synonym)
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
    
    return expanded_queries[:3]  # 최대 3개 쿼리로 제한

# 초기화 함수
def initialize_rag_system():
    """RAG 시스템의 모든 구성 요소를 초기화합니다."""
    logger.info("RAG 시스템 초기화 중...")
    
    # 임베딩 모델 설정
    embeddings_model = get_embeddings()
    if embeddings_model is None:
        logger.error("임베딩 모델을 초기화할 수 없습니다.")
        return None, None, None
    
    # 파일 경로 설정
    data_file = "narosu_db_final.csv"
    excel_file = "narosu_db_final.xlsx"
    embeddings_dir = "embeddings_checkpoints"
    
    # 데모 모드 확인
    demo_mode = check_demo_mode(file_path=data_file, embeddings_dir=embeddings_dir)
    

    # 상품 데이터 로드
    product_data = load_product_data(file_path=data_file, excel_path=excel_file)
    
    # Qdrant 클라이언트 설정
    collection_name = "product_recommendations"
    qdrant_client = setup_qdrant_client(collection_name)
    
    # 임베딩 데이터 로드
    embeddings_data, product_ids = load_embeddings(embeddings_dir=embeddings_dir)
    

    else:
        # 벡터 저장소 초기화
        vector_store = init_vector_store(qdrant_client, collection_name, product_data, embeddings_data, product_ids, embeddings_model)
    
    # LLM 초기화
    llm = init_llm()
    
    # RAG 파이프라인 초기화
    if vector_store is not None and llm is not None:
        rag_pipeline = init_rag_pipeline(vector_store, llm)
    else:
        logger.error("벡터 저장소 또는 LLM을 초기화할 수 없습니다.")
        rag_pipeline = None
    
    # 채팅 기록 초기화
    chat_history = ChatHistory()
    
    if rag_pipeline is not None:
        logger.info("RAG 시스템 초기화 완료!")
    else:
        logger.error("RAG 시스템 초기화 실패!")
    
    return rag_pipeline, chat_history, vector_store

# 챗봇 응답 생성 함수
def generate_response(message, rag_pipeline, chat_history, vector_store):
    """사용자 메시지에 대한 응답을 생성합니다."""
    if not message:
        return "메시지를 입력해주세요."
    
    if rag_pipeline is None:
        return "RAG 시스템이 초기화되지 않았습니다. 관리자에게 문의하세요."
    
    try:
        # 메시지를 채팅 기록에 추가
        chat_history.add_message("user", message)
        
        # 쿼리 전처리
        clean_query = preprocess_query(message)
        
        # 네이버 쇼핑 API 호출
        naver_products = search_naver_shopping(clean_query)
        formatted_naver_products = format_naver_products(naver_products)
        
        # RAG 검색 쿼리 실행
        prompt = chat_history.format_prompt(clean_query)
        result = rag_pipeline({"query": clean_query})
        
        # 검색 기록 저장
        chat_history.add_search_result(clean_query, result.get('source_documents', []))
        
        # 검색 결과 포맷팅
        response = format_search_results(result)
        
        # 네이버 쇼핑 결과 추가
        if formatted_naver_products:
            response += "\n\n📦 네이버 쇼핑 추천 상품:\n\n" + "\n\n".join(formatted_naver_products)
        
        # 상품 가격 분포 차트 생성 (조건부)
        if "가격" in message.lower() or "얼마" in message.lower():
            price_chart = create_price_distribution_chart(result.get('source_documents', []))
            if price_chart:
                response += f"\n\n### 검색 결과 가격 분포\n{price_chart}"
        
        # 응답을 채팅 기록에 추가
        chat_history.add_message("assistant", response)
        
        return response
    except Exception as e:
        logger.error(f"응답 생성 중 오류 발생: {e}")
        error_msg = f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 오류: {str(e)}"
        chat_history.add_message("assistant", error_msg)
        return error_msg
    
    
# 검색 통계 시각화 함수
def create_search_analytics(chat_history):
    """검색 통계를 텍스트로 반환합니다."""
    analytics = chat_history.get_search_analytics()
    
    if isinstance(analytics, str):
        return analytics
    
    # 검색 통계 포맷팅
    result = "## 🔍 검색 통계\n\n"
    result += f"총 검색 횟수: {analytics['total_searches']}회\n\n"
    
    if analytics['top_categories']:
        result += "### 인기 카테고리:\n"
        for i, (category, count) in enumerate(analytics['top_categories'][:5], 1):
            result += f"{i}. {category}: {count}회\n"
        result += "\n"
    
    if analytics['search_queries']:
        result += "### 최근 검색어:\n"
        recent_queries = analytics['search_queries'][-5:]  # 최근 5개 검색어
        for i, query in enumerate(recent_queries, 1):
            result += f"{i}. {query}\n"
    
    return result

import urllib.request
import json

def search_naver_shopping(query, display=2):
    """네이버 쇼핑 API를 사용하여 상품을 검색합니다."""
    try:
        client_id = "AQncjiAhGcWf0WegURyt"
        client_secret = "XFVLBoNQxM"
        
        # 검색어 인코딩
        encoded_query = urllib.parse.quote(query)
        url = f"https://openapi.naver.com/v1/search/shop.json?query={encoded_query}&display={display}&sort=sim"
        
        # API 요청 생성
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        
        # API 호출 및 응답 처리
        response = urllib.request.urlopen(request)
        response_code = response.getcode()
        
        if response_code == 200:
            response_body = response.read()
            response_data = json.loads(response_body.decode('utf-8'))
            return response_data.get('items', [])
        else:
            logger.error(f"네이버 API 요청 실패: 응답 코드 {response_code}")
            return []
    except Exception as e:
        logger.error(f"네이버 쇼핑 API 호출 중 오류 발생: {e}")
        return []

def format_naver_products(products):
    """네이버 쇼핑 API 결과를 포맷팅합니다."""
    formatted_products = []
    
    for i, product in enumerate(products, 1):
        title = product.get('title', '').replace('<b>', '').replace('</b>', '')
        price = int(product.get('lprice', '0'))
        mall_name = product.get('mallName', '알 수 없음')
        product_id = product.get('productId', '')
        product_type = product.get('productType', '')
        link = product.get('link', '')
        image = product.get('image', '')
        category = product.get('category1', '') + ' > ' + product.get('category2', '')
        
        product_info = f"【네이버 추천 {i}】 {title}\n"
        product_info += f"   📂 카테고리: {category}\n"
        product_info += f"   💰 가격: {price:,}원\n"
        product_info += f"   🏬 판매처: {mall_name}\n"
        product_info += f"   🔗 링크: {link}\n"
        
        formatted_products.append(product_info)
    
    return formatted_products


# respond 함수 (예시 질문 특별 처리)
def respond(message, history):
    """사용자 메시지에 대한 응답을 생성합니다."""
    if message.strip() == "":
        return "", history
    
    
    # 특수 명령어 처리
    if message.strip() == "/통계":
        try:
            stats = create_search_analytics(chat_history)
            history.append((message, stats))
            return "", history
        except Exception as e:
            error_msg = f"통계 생성 중 오류 발생: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    # 일반 응답 생성
    try:
        if rag_pipeline is None:
            error_msg = "죄송합니다. RAG 시스템이 초기화되지 않았습니다."
            history.append((message, error_msg))
        else:
            # 채팅 기록에 사용자 메시지 추가
            chat_history.add_message("user", message)
            
            # 쿼리 전처리 및 RAG 검색
            clean_query = preprocess_query(message)
            result = rag_pipeline({"query": clean_query})
            
            # 결과 포맷팅 (이미지 포함)
            response = format_search_results(result)
            
            # 채팅 기록에 응답 추가
            chat_history.add_message("assistant", response)
            
            # 히스토리에 추가
            history.append((message, response))
        
        return "", history
    except Exception as e:
        logger.error(f"응답 생성 중 오류 발생: {e}", exc_info=True)
        error_msg = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
        history.append((message, error_msg))
        return "", history
    

def create_gradio_interface():
    """Gradio 채팅 인터페이스를 생성합니다."""
    # 파일 경로 확인 로깅
    current_dir = os.getcwd()
    logger.info(f"현재 작업 디렉토리: {current_dir}")
    
    data_file = "narosu_db_final.csv"
    excel_file = "narosu_db_final.xlsx"
    embeddings_dir = "embeddings_checkpoints"
    checkpoint_file = os.path.join(embeddings_dir, "embeddings_checkpoint.pkl")
    
    logger.info(f"CSV 파일 경로: {os.path.join(current_dir, data_file)}, 존재 여부: {os.path.exists(data_file)}")
    logger.info(f"Excel 파일 경로: {os.path.join(current_dir, excel_file)}, 존재 여부: {os.path.exists(excel_file)}")
    logger.info(f"임베딩 디렉토리: {os.path.join(current_dir, embeddings_dir)}, 존재 여부: {os.path.exists(embeddings_dir)}")
    logger.info(f"체크포인트 파일: {os.path.join(current_dir, checkpoint_file)}, 존재 여부: {os.path.exists(checkpoint_file)}")
    
    # RAG 시스템 초기화
    global rag_pipeline, chat_history, vector_store
    rag_pipeline, chat_history, vector_store = initialize_rag_system()
    
    # 인터페이스 생성
    with gr.Blocks(css="footer {visibility: hidden}") as interface:
        gr.Markdown("# 🛍️ 스마트 상품 검색 어시스턴트")
        gr.Markdown("상품에 관한 질문이나 추천을 요청해보세요. 아래 예시 질문을 클릭하거나 직접 질문을 입력하세요.")
        
        chatbot = gr.Chatbot(
            height=500,
            bubble_full_width=False,
            show_label=False,
            show_copy_button=True,
            layout="panel"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="상품에 관해 무엇이든 물어보세요...",
                scale=9,
                show_label=False,
                container=False
            )
            submit = gr.Button("전송", scale=1, variant="primary")
        
        # 예시 질문 섹션 추가
        gr.Markdown("## 💡 바로 사용해볼 수 있는 예시 질문")
        
        with gr.Row():
            example_btn1 = gr.Button("여성용 가을 자켓 추천해줘", scale=1)
            example_btn2 = gr.Button("2만원 이하의 주방용품 알려줘", scale=1)
            example_btn3 = gr.Button("아이폰 15 케이스 추천", scale=1)
        
        with gr.Row():
            example_btn4 = gr.Button("검은색 크로스백 추천해줘", scale=1)
            example_btn5 = gr.Button("요즘 트렌드 패션 아이템은?", scale=1)
            example_btn6 = gr.Button("/통계", scale=1)
        
        # 이벤트 핸들링
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        
        # 예시 버튼에 이벤트 연결
        example_btn1.click(lambda: "여성용 가을 자켓 추천해줘", None, msg).then(
            respond, [msg, chatbot], [msg, chatbot]
        )
        example_btn2.click(lambda: "2만원 이하의 주방용품 알려줘", None, msg).then(
            respond, [msg, chatbot], [msg, chatbot]
        )
        example_btn3.click(lambda: "아이폰 15 케이스 추천", None, msg).then(
            respond, [msg, chatbot], [msg, chatbot]
        )
        example_btn4.click(lambda: "검은색 크로스백 추천해줘", None, msg).then(
            respond, [msg, chatbot], [msg, chatbot]
        )
        example_btn5.click(lambda: "요즘 트렌드 패션 아이템은?", None, msg).then(
            respond, [msg, chatbot], [msg, chatbot]
        )
        example_btn6.click(lambda: "/통계", None, msg).then(
            respond, [msg, chatbot], [msg, chatbot]
        )
        
        # 추가 정보 섹션
        with gr.Accordion("도움말 및 추가 예시 질문", open=False):
            gr.Markdown("""
            ### 🔍 이런 질문도 물어보세요:
            
            - 가을용 여성 자켓 중에 가장 인기있는 것은?
            - 5만원 이하의 선물용 주방용품 추천해줘
            - 캐주얼한 스타일의 남성 의류 알려줘
            - 겨울철 따뜻한 의류 추천해주세요
            - 가성비 좋은 주방 용품은?
            - 디지털 제품 중 인기 있는 것
            
            ### 💡 명령어 안내
            - **/통계**: 검색 통계 확인하기
            
            ### 🛍️ 상품 검색 팁
            - 가격대를 구체적으로 언급하면 더 정확한 추천을 받을 수 있습니다
            - 색상, 스타일, 용도 등 구체적인 조건을 포함하면 좋습니다
            - 계절, 연령대 등을 명시하면 맞춤형 추천이 가능합니다
            """)
    
    return interface



# 메인 함수
def main():
    """메인 애플리케이션 실행 함수"""
    interface = create_gradio_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()