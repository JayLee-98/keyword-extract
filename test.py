# test.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Any
import re

class AISearchItem(BaseModel):
    key: int
    type: str = ""  # T:title, S:summary, K:keyword
    text: str = ""

class AIChunk(BaseModel):
    title: str = ""
    summary: str = ""
    keyword_list: List[str] = []
    embbeding: Any = Field(sa_column=Column(Vector(1024)))

class AIResult(BaseModel):
    title: str = ""
    chunc_list: List[AIChunk] = []

class AIResultContents(BaseModel):
    source: str = ""
    result: AIResult = AIResult()

class TextAnalyzer:
    def __init__(self):
        # BERT 모델 초기화
        self.bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.bert_model = AutoModel.from_pretrained("klue/bert-base")



# Python Code


tokenizer = AutoTokenizer.from_pretrained("noahkim/KoT5_news_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("noahkim/KoT5_news_summarization")

        
        # BART 모델 초기화 
        self.bart_tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
        self.bart_summarizer = pipeline(
            "summarization",
            model="gogamza/kobart-base-v2",
            tokenizer=self.bart_tokenizer
        )

    def split_into_chunks(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """
        텍스트를 의미 단위로 청크로 분리
        1. 문장 단위로 먼저 분리
        2. 임베딩 기반으로 유사한 문장들을 군집화
        """
        # 문장 단위로 분리
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 문장별 임베딩 생성
        embeddings = []
        for sentence in sentences:
            inputs = self.bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding[0].numpy())
        
        # KMeans 클러스터링으로 유사한 문장들을 그룹화
        n_clusters = max(1, len(sentences) // 3)  # 청크 크기 조정 가능
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)
        
        # 클러스터별로 청크 생성
        chunks = []
        for i in range(n_clusters):
            chunk_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
            chunks.append(" ".join(chunk_sentences))
            
        return chunks

    def generate_title(self, text: str) -> str:
        """BART를 사용하여 청크의 제목 생성"""
        summary = self.bart_summarizer(text, max_length=20, min_length=5, do_sample=False)[0]['summary_text']
        return summary.strip()

    def generate_summary(self, text: str) -> str:
        """BART를 사용하여 청크의 요약 생성"""
        summary = self.bart_summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return summary.strip()

    def extract_keywords(self, text: str) -> List[str]:
        """BERT를 사용하여 키워드 추출"""
        # 토큰화 및 임베딩 생성
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]
            
        # 토큰별 중요도 계산
        importance_scores = torch.norm(token_embeddings, dim=1)
        
        # 상위 N개 토큰 선택
        n_keywords = min(10, len(inputs['input_ids'][0]))
        top_indices = torch.argsort(importance_scores, descending=True)[:n_keywords]
        
        # 토큰을 다시 텍스트로 변환
        keywords = []
        for idx in top_indices:
            token = self.bert_tokenizer.decode([inputs['input_ids'][0][idx]])
            if token.strip() and not token.startswith('##'):
                keywords.append(token.strip())
                
        return list(set(keywords))  # 중복 제거

    def get_embedding(self, text: str) -> np.ndarray:
        """텍스트의 임베딩 벡터 생성"""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # [CLS] 토큰의 임베딩을 문서 벡터로 사용
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding[0]

async def request_ai_result(source_text: str, source_title: Optional[str] = None) -> AIResultContents:
    analyzer = TextAnalyzer()
    
    # 텍스트를 청크로 분리
    chunks = analyzer.split_into_chunks(source_text)
    
    # 각 청크에 대한 분석 수행
    chunk_list = []
    for chunk_text in chunks:
        chunk = AIChunk(
            title=analyzer.generate_title(chunk_text),
            summary=analyzer.generate_summary(chunk_text),
            keyword_list=analyzer.extract_keywords(chunk_text),
            embbeding=analyzer.get_embedding(chunk_text)
        )
        chunk_list.append(chunk)
    
    # 전체 제목 생성 또는 주어진 제목 사용
    title = source_title if source_title else analyzer.generate_title(source_text)
    
    # 결과 반환
    result = AIResult(title=title, chunc_list=chunk_list)
    return AIResultContents(source=source_text, result=result)

async def request_at_keyword_list(source_text: str) -> List[str]:
    analyzer = TextAnalyzer()
    return analyzer.extract_keywords(source_text)

async def request_at_embbeding(source_text: str) -> np.ndarray:
    analyzer = TextAnalyzer()
    return analyzer.get_embedding(source_text)