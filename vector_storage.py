"""#
벡터 저장소 모듈
ChromaDB 기반 장기 기억 벡터 저장소
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from memory_entry import MemoryEntry

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("경고: ChromaDB 또는 SentenceTransformer를 임포트할 수 없습니다. 일부 기능이 제한됩니다.")


class VectorStorage:
    """
    ChromaDB 기반 벡터 저장소 클래스
    
    특징:
    - 장기 기억 관리
    - 의미 기반 유사성 검색
    - 계층적 메모리 저장
    """
    
    def __init__(self, 
                 db_path: str = "./chroma_db", 
                 collection_name: str = "long_term_memories",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        벡터 저장소 초기화
        
        Args:
            db_path: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델명
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # 의존성 체크
        if not HAS_DEPENDENCIES:
            self.client = None
            self.collection = None
            self.model = None
            print("경고: 필요한 라이브러리가 설치되지 않아 VectorStorage가 제한됩니다.")
            return
        
        # ChromaDB 초기화
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 컬렉션 가져오기 또는 생성
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
        
        # 임베딩 모델 로드
        try:
            self.model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"임베딩 모델 로드 오류: {e}")
            self.model = None
        
        # 통계
        self.stats = {
            'save_count': 0,
            'search_count': 0,
            'total_memories': 0
        }
    
    async def save(self, memory: MemoryEntry) -> str:
        """
        메모리 저장
        
        Args:
            memory: 저장할 메모리 엔트리
            
        Returns:
            저장된 메모리 ID
        """
        if not HAS_DEPENDENCIES or not self.collection or not self.model:
            print("경고: 필요한 라이브러리가 설치되지 않아 저장을 건너뜁니다.")
            return memory.id
        
        # 컨텐츠에서 텍스트 추출
        text_repr = self._extract_text_from_content(memory.content)
        
        # 개념 목록을 텍스트로 변환
        concepts_text = " ".join(memory.concepts)
        
        # 메타데이터 준비
        metadata = {
            'id': memory.id,
            'concepts': json.dumps(memory.concepts),
            'importance': memory.importance,
            'emotional_weight': memory.emotional_weight,
            'creation_time': memory.creation_time.isoformat()
        }
        
        # 메모리 내용을 JSON으로 직렬화
        serialized_memory = json.dumps(memory.to_dict())
        
        # 임베딩 생성
        try:
            # 임베딩 생성은 동기 작업이므로 블로킹 가능성 있음
            embedding = self.model.encode(text_repr + " " + concepts_text).tolist()
            
            # 컬렉션에 저장
            self.collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[serialized_memory]
            )
            
            # 통계 업데이트
            self.stats['save_count'] += 1
            self.stats['total_memories'] += 1
            
            return memory.id
        except Exception as e:
            print(f"벡터 저장소 저장 오류: {e}")
            return memory.id
    
    def _extract_text_from_content(self, content: Dict[str, Any]) -> str:
        """
        컨텐츠에서 텍스트 추출
        
        Args:
            content: 메모리 컨텐츠
            
        Returns:
            텍스트 표현
        """
        text_parts = []
        
        # 사용자 및 어시스턴트 메시지 추가
        if 'user' in content:
            text_parts.append(f"User: {content['user']}")
        if 'assistant' in content:
            text_parts.append(f"Assistant: {content['assistant']}")
        
        # 메모 필드 추가
        if 'memo' in content:
            text_parts.append(f"Memo: {content['memo']}")
        
        # 토픽 및 서브토픽 추가
        if 'topic' in content:
            text_parts.append(f"Topic: {content['topic']}")
        if 'sub_topic' in content:
            text_parts.append(f"Subtopic: {content['sub_topic']}")
        
        # 기타 문자열 필드 추가
        for key, value in content.items():
            if isinstance(value, str) and key not in ['user', 'assistant', 'memo', 'topic', 'sub_topic']:
                text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)
    
    async def search_by_concepts(self, concepts: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        개념 기반 검색
        
        Args:
            concepts: 검색할 개념 목록
            top_k: 반환할 최대 결과 수
            
        Returns:
            검색 결과 목록
        """
        if not HAS_DEPENDENCIES or not self.collection or not self.model:
            print("경고: 필요한 라이브러리가 설치되지 않아 검색을 건너뜁니다.")
            return []
        
        if not concepts:
            return []
        
        try:
            # 개념을 텍스트로 변환하여 쿼리 생성
            query_text = " ".join(concepts)
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode(query_text).tolist()
            
            # 유사도 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 검색 결과 가공
            memory_results = []
            
            if 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    # 메모리 복원
                    try:
                        memory_dict = json.loads(doc)
                        memory = MemoryEntry.from_dict(memory_dict)
                        
                        # 유사도 점수
                        similarity = 1.0 - (i / len(results['documents'][0]))
                        
                        memory_results.append({
                            'memory': memory,
                            'similarity': similarity
                        })
                    except Exception as e:
                        print(f"메모리 복원 오류: {e}")
            
            # 통계 업데이트
            self.stats['search_count'] += 1
            
            return memory_results
        except Exception as e:
            print(f"벡터 검색 오류: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        벡터 저장소 통계 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        if HAS_DEPENDENCIES and self.collection:
            # 실제 메모리 수 업데이트
            try:
                collection_info = self.collection.count()
                self.stats['total_memories'] = collection_info
            except:
                pass
        
        return {
            'total_memories': self.stats['total_memories'],
            'save_count': self.stats['save_count'],
            'search_count': self.stats['search_count'],
            'available': HAS_DEPENDENCIES
        }