"""#
Neo4j 설정 관리 모듈
Neo4j 기반 시스템의 설정값들을 중앙에서 관리
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Neo4jSystemConfig:
    """Neo4j 시스템 전반의 설정"""
    # Neo4j 설정
    neo4j_uri: str = 'neo4j://localhost:7687'
    neo4j_user: str = 'neo4j'
    neo4j_password: str = 'password'
    
    # ChromaDB 설정 (장기 기억용)
    chroma_path: str = './chroma_db'
    collection_name: str = 'long_term_memories'
    
    # 메모리 관리 설정
    max_active_memory: int = 1000
    max_session_duration: int = 3600  # 1시간
    
    # 연관 네트워크 설정
    min_connection_strength: float = 0.1
    decay_factor: float = 0.95
    connection_strength_threshold: float = 0.5
    
    # 임베딩 모델 설정
    embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'neo4j': {
                'uri': self.neo4j_uri,
                'user': self.neo4j_user,
                'password': self.neo4j_password
            },
            'storage': {
                'chroma_path': self.chroma_path,
                'collection_name': self.collection_name
            },
            'memory': {
                'max_active_memory': self.max_active_memory,
                'max_session_duration': self.max_session_duration
            },
            'network': {
                'min_connection_strength': self.min_connection_strength,
                'decay_factor': self.decay_factor,
                'connection_strength_threshold': self.connection_strength_threshold
            },
            'embedding': {
                'model': self.embedding_model
            }
        }
