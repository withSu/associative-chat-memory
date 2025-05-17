"""#
Neo4j 기반 메모리 관리 모듈
Neo4j 그래프 데이터베이스를 사용한 메모리 관리
"""
from utils import LRUCache, BloomFilter  
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from memory_entry import MemoryEntry  # 또는 .models.memory_entry
from enums import MemoryTier, ConnectionType  # 또는 .models.enums
from neo4j_storage import Neo4jStorage
from vector_storage import VectorStorage  # vector_storage.py 파일을 루트로 옮기거나 경로 수정

class Neo4jMemoryManager:
    """
    Neo4j 기반 메모리 관리자
    
    기능:
    - Neo4j 그래프 기반 메모리 저장
    - 그래프 기반 연관 검색
    - 연결 강도 기반 장기 기억 승격
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Neo4j 저장소 초기화
        self.neo4j = Neo4jStorage(
            uri=config['neo4j']['uri'],
            user=config['neo4j']['user'],
            password=config['neo4j']['password'],
            min_connection_strength=config['network'].get('min_connection_strength', 0.1),
            connection_strength_threshold=config['network'].get('connection_strength_threshold', 0.5)
        )
        
        # 장기 기억 벡터 저장소
        self.vector = VectorStorage(
            db_path=config['storage']['chroma_path'],
            collection_name=config['storage']['collection_name'],
            embedding_model=config['embedding']['model']
        )
        
        # 캐시 및 인덱스
        self.search_cache = LRUCache(capacity=1000)
        self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)
        
        # 설정값
        self.connection_strength_threshold = config['network'].get('connection_strength_threshold', 0.5)
        self.promotion_threshold = config['network'].get('promotion_threshold', 0.8)
        
        # 통계
        self.stats = {
            'total_memories': 0,
            'search_operations': 0,
            'save_operations': 0,
            'tier_distribution': {tier.value: 0 for tier in MemoryTier}
        }
    
    async def save_memory(
        self,
        content: Dict[str, Any],
        concepts: List[str],
        importance: float = 0.5,
        emotional_weight: float = 0.0,
        tier: MemoryTier = MemoryTier.SHORT_TERM
    ) -> str:
        """메모리 저장"""
        memory_id = str(uuid.uuid4())
        
        # 메모리 엔트리 생성
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            concepts=concepts,
            importance=importance,
            emotional_weight=emotional_weight,
            tier=tier
        )
        
        # 티어에 따라 적절한 저장소에 저장
        if tier == MemoryTier.SHORT_TERM:
            # Neo4j에 저장
            await self.neo4j.save_memory(memory)
        elif tier == MemoryTier.LONG_TERM:
            # 벡터 저장소에 저장
            await self.vector.save(memory)
        
        # 블룸 필터 업데이트
        for concept in concepts:
            self.bloom_filter.add(concept)
        
        # 통계 업데이트
        self.stats['total_memories'] += 1
        self.stats['save_operations'] += 1
        self.stats['tier_distribution'][tier.value] += 1
        
        return memory_id
    
    async def process_external_keywords(
        self,
        keywords: List[str],
        content: Dict[str, Any],
        importance: float = 0.5,
        emotional_weight: float = 0.0
    ) -> str:
        """
        외부에서 받은 키워드를 처리하고 적절한 메모리에 저장
        
        Args:
            keywords: 외부에서 전달받은 키워드 목록
            content: 저장할 컨텐츠 데이터
            importance: 중요도 점수
            emotional_weight: 감정 가중치
            
        Returns:
            저장된 메모리 ID
        """
        # 키워드(컨셉트)들을 대상으로 메모리 저장
        return await self.save_memory(
            content=content,
            concepts=keywords,
            importance=importance,
            emotional_weight=emotional_weight,
            tier=MemoryTier.SHORT_TERM
        )
    
    async def search_memories(
        self,
        concepts: List[str],
        query_text: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """통합 메모리 검색"""
        # 캐시 확인
        cache_key = f"search:{':'.join(concepts)}"
        cached_result = self.search_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 병렬 검색 실행
        search_tasks = [
            self.neo4j.search_memories(concepts, query_text, top_k),
            self.vector.search_by_concepts(concepts, top_k)
        ]
        
        results = await asyncio.gather(*search_tasks)
        
        # 결과 병합 및 재순위
        neo4j_results = results[0]
        vector_results = results[1]
        
        all_results = []
        
        # Neo4j 결과 처리
        for item in neo4j_results:
            all_results.append(item)  # 이미 적절한 형식
        
        # Vector DB 결과 처리
        for result in vector_results:
            score = 0.6 * result['similarity']
            all_results.append({
                'memory': result,
                'score': score,
                'source': 'long_term'
            })
        
        # 점수 기반 정렬
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        
        # 중복 제거
        unique_results = []
        seen_ids = set()
        
        for result in sorted_results:
            memory_id = result['memory'].id if hasattr(result['memory'], 'id') else result['memory']['id']
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_results.append(result)
        
        # 캐시에 저장
        self.search_cache.put(cache_key, unique_results[:top_k])
        
        # 통계 업데이트
        self.stats['search_operations'] += 1
        
        return unique_results[:top_k]
    
    async def save_topic_subtopic_relations(self, json_data: List[Dict[str, str]]) -> None:
        """
        JSON 데이터에서 추출한 토픽과 서브토픽 관계를 저장
        
        Args:
            json_data: 친구 시스템에서 제공하는 JSON 데이터
        """
        # Neo4j에 토픽/서브토픽 관계 저장
        await self.neo4j.process_json_keywords(json_data)
    
    async def get_weak_connections(self, threshold: float = None) -> List[Dict[str, Any]]:
        """
        약한 연결 조회
        
        Args:
            threshold: 연결 강도 임계값 (이 값 이하의 연결이 조회됨)
            
        Returns:
            약한 연결 목록
        """
        return await self.neo4j.get_weak_connections(threshold)
    
    async def promote_weak_connections_to_long_term(self, threshold: float = None) -> int:
        """
        약한 연결 강도를 가진 개념들의 메모리를 장기 기억으로 승격
        
        Args:
            threshold: 약한 연결로 간주할 임계값 (없으면 기본값 사용)
            
        Returns:
            승격된 메모리 수
        """
        return await self.neo4j.promote_weak_connections_to_long_term(threshold)
    
    async def check_for_promotion_to_long_term(self, concept: str) -> None:
        """
        특정 개념을 장기 기억으로 승격할지 검토
        
        Args:
            concept: 검토할 개념
        """
        # Neo4j는 자체적으로 처리하므로 직접 호출
        await self.neo4j.promote_weak_connections_to_long_term([concept])
    
    async def visualize_graph(self, center_concept: Optional[str] = None) -> Dict[str, Any]:
        """
        그래프 시각화 데이터 추출
        
        Args:
            center_concept: 중심 개념 (없으면 전체 그래프)
            
        Returns:
            시각화 데이터 (NetworkX와 호환되는 형식)
        """
        return await self.neo4j.visualize_graph(center_concept)
    
    async def get_stats(self) -> Dict[str, Any]:
        """통합 통계 반환"""
        # Neo4j 통계
        neo4j_stats = await self.neo4j.get_stats()
        
        # Vector DB 통계
        vector_stats = self.vector.get_stats()
        
        return {
            'total_memories': self.stats['total_memories'],
            'tier_distribution': {
                'short_term': neo4j_stats['neo4j']['total_memories'],
                'long_term': vector_stats['total_memories']
            },
            'operations': {
                'search': self.stats['search_operations'],
                'save': self.stats['save_operations']
            },
            'resources': {
                'cache_size': len(self.search_cache.cache)
            },
            'neo4j': neo4j_stats['neo4j'],
            'vector': vector_stats
        }
    
    async def apply_decay(self, decay_factor: float = 0.95) -> None:
        """연결 강도 감소 적용"""
        await self.neo4j.apply_decay(decay_factor)
    
    async def shutdown(self) -> None:
        """시스템 종료"""
        # Neo4j 연결 종료
        self.neo4j.close()