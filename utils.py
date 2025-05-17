"""#
유틸리티 클래스 모듈
메모리 관리와 네트워크 분석을 위한 도구들
"""
from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple
import random
from datetime import datetime


class LRUCache:
    """
    LRU(Least Recently Used) 캐시 구현
    최근에 사용되지 않은 항목을 우선적으로 제거하는 캐시
    """
    
    def __init__(self, capacity: int = 1000):
        """
        LRU 캐시 초기화
        
        Args:
            capacity: 최대 캐시 크기
        """
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Any:
        """
        캐시에서 항목 조회
        
        Args:
            key: 조회할 키
        
        Returns:
            저장된 값 또는 None (없는 경우)
        """
        if key not in self.cache:
            return None
        
        # 접근한 항목을 맨 뒤로 이동 (최근 사용)
        value = self.cache.pop(key)
        self.cache[key] = value
        
        return value
    
    def put(self, key: str, value: Any) -> None:
        """
        캐시에 항목 저장
        
        Args:
            key: 키
            value: 저장할 값
        """
        # 이미 존재하는 키라면 제거 후 재삽입
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # 캐시가 가득 찬 경우 가장 오래된 항목 제거
            self.cache.popitem(last=False)
        
        self.cache[key] = value


class BloomFilter:
    """
    블룸 필터 구현
    효율적인 중복 검사를 위한 확률적 자료구조
    """
    
    def __init__(self, capacity: int = 100000, error_rate: float = 0.001):
        """
        블룸 필터 초기화
        
        Args:
            capacity: 예상 항목 수
            error_rate: 허용 오류율
        """
        self.capacity = capacity
        self.error_rate = error_rate
        
        # 비트 배열 크기 계산
        # m = -(n * ln(p)) / (ln(2)^2)
        self.size = int(-(capacity * 2.303 * error_rate) / 0.4804)
        
        # 최적 해시 함수 수 계산
        # k = (m/n) * ln(2)
        self.hash_count = int((self.size / capacity) * 0.693)
        
        # 비트 배열 초기화
        self.bit_array = [0] * self.size
    
    def _hash_functions(self, item: str) -> List[int]:
        """
        여러 해시 함수 결과 생성
        
        Args:
            item: 해시할 항목
        
        Returns:
            해시 함수 결과 인덱스 목록
        """
        # 간단한 해시 함수 구현 (실제로는 더 효율적인 해시 함수 사용 필요)
        hash_indices = []
        
        # 기본 해시값
        hash1 = hash(item)
        hash2 = hash(item + item)
        
        for i in range(self.hash_count):
            # 선형 결합 기반 해시 함수
            index = (hash1 + i * hash2) % self.size
            hash_indices.append(index)
        
        return hash_indices
    
    def add(self, item: str) -> None:
        """
        블룸 필터에 항목 추가
        
        Args:
            item: 추가할 항목
        """
        for index in self._hash_functions(item):
            self.bit_array[index] = 1
    
    def contains(self, item: str) -> bool:
        """
        항목이 블룸 필터에 있는지 확인
        
        Args:
            item: 확인할 항목
            
        Returns:
            포함 여부 (False: 확실히 없음, True: 있을 가능성 있음)
        """
        for index in self._hash_functions(item):
            if not self.bit_array[index]:
                return False
        return True


class AssociationNetwork:
    """
    연관 네트워크 관리
    개념 간 연결 및 활성화 관리
    """
    
    def __init__(self, min_strength: float = 0.1, decay_factor: float = 0.95):
        """
        연관 네트워크 초기화
        
        Args:
            min_strength: 최소 연결 강도 (이 값 이하의 연결은 제거)
            decay_factor: 감쇠 계수 (시간에 따른 연결 약화)
        """
        # 개념 노드 저장소
        self.concepts = {}
        
        # 개념 간 연결 저장소 (source → target → 강도)
        self.connections = {}
        
        # 설정값
        self.min_strength = min_strength
        self.decay_factor = decay_factor
        
        # 통계
        self.activations = 0
        self.new_connections = 0
    
    def activate_concept(self, concept: str, related_concepts: List[str]) -> None:
        """
        개념 활성화 및 관련 개념과의 연결 생성/강화
        
        Args:
            concept: 활성화할 개념
            related_concepts: 함께 활성화된 관련 개념 목록
        """
        # 개념이 없으면 생성
        if concept not in self.concepts:
            self.concepts[concept] = {
                'activation_count': 0,
                'last_activated': datetime.now(),
                'importance': 0.5
            }
        
        # 활성화 정보 업데이트
        self.concepts[concept]['activation_count'] += 1
        self.concepts[concept]['last_activated'] = datetime.now()
        
        # 관련 개념과의 연결 생성/강화
        for related in related_concepts:
            if related != concept:
                self.connect_concepts(concept, related)
        
        self.activations += 1
    
    def connect_concepts(self, source: str, target: str, strength: float = None, connection_type: str = 'semantic') -> None:
        """
        두 개념 간 연결 생성/강화
        
        Args:
            source: 소스 개념
            target: 타겟 개념
            strength: 연결 강도 (명시적 지정 시)
            connection_type: 연결 유형
        """
        # 양쪽 개념이 존재하는지 확인
        for c in [source, target]:
            if c not in self.concepts:
                self.concepts[c] = {
                    'activation_count': 0,
                    'last_activated': datetime.now(),
                    'importance': 0.5
                }
        
        # source → connections 딕셔너리 초기화
        if source not in self.connections:
            self.connections[source] = {}
        
        # 이미 연결이 있는 경우 강화, 없는 경우 생성
        if target in self.connections[source]:
            if strength is not None:
                # 명시적 강도 설정
                self.connections[source][target]['strength'] = strength
            else:
                # 기존 강도 증가 (최대 0.9)
                curr_strength = self.connections[source][target]['strength']
                self.connections[source][target]['strength'] = min(curr_strength + 0.1, 0.9)
            
            self.connections[source][target]['updated_at'] = datetime.now()
            self.connections[source][target]['access_count'] += 1
        else:
            # 새 연결 생성
            self.connections[source][target] = {
                'strength': strength or 0.5,  # 기본 강도는 0.5
                'type': connection_type,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'access_count': 1
            }
            self.new_connections += 1
    
    def find_associations(self, concepts: List[str], top_k: int = 5) -> Dict[str, float]:
        """
        주어진 개념들과 연관성이 높은 다른 개념 찾기
        
        Args:
            concepts: 기준 개념 목록
            top_k: 반환할 연관 개념 수
            
        Returns:
            연관 개념:점수 딕셔너리
        """
        associations = {}
        
        # 각 개념에 대해 연관된 개념 찾기
        for concept in concepts:
            if concept in self.connections:
                for target, conn_data in self.connections[concept].items():
                    if target not in concepts:  # 이미 기준 목록에 있는 개념은 제외
                        strength = conn_data['strength']
                        
                        # 강도에 따라 점수 계산
                        if target in associations:
                            associations[target] = max(associations[target], strength)
                        else:
                            associations[target] = strength
        
        # 점수 기준 정렬하여 상위 k개 반환
        sorted_associations = dict(
            sorted(associations.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        
        return sorted_associations
    
    def apply_decay(self) -> int:
        """
        시간에 따른 연결 강도 감쇠 적용
        
        Returns:
            제거된 연결 수
        """
        to_remove = []
        removed_count = 0
        
        # 모든 연결에 감쇠 적용
        for source, targets in self.connections.items():
            for target, conn_data in list(targets.items()):
                # 강도 감쇠
                conn_data['strength'] *= self.decay_factor
                
                # 최소 강도 이하면 제거 목록에 추가
                if conn_data['strength'] < self.min_strength:
                    to_remove.append((source, target))
        
        # 약한 연결 제거
        for source, target in to_remove:
            del self.connections[source][target]
            removed_count += 1
            
            # 연결이 없어진 소스는 딕셔너리에서 제거
            if not self.connections[source]:
                del self.connections[source]
        
        return removed_count
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        네트워크 통계 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        # 총 연결 수 계산
        total_connections = sum(len(targets) for targets in self.connections.values())
        
        # 강한 연결 수 계산
        strong_connections = sum(
            1 for source in self.connections
            for target in self.connections[source]
            if self.connections[source][target]['strength'] >= 0.7
        )
        
        return {
            'total_concepts': len(self.concepts),
            'total_connections': total_connections,
            'strong_connections': strong_connections,
            'weak_connections': total_connections - strong_connections,
            'activations': self.activations,
            'new_connections': self.new_connections
        }


class LifecycleManager:
    """
    메모리 생명주기 관리
    메모리 승격, 감쇠 등의 주기적 작업 관리
    """
    
    def __init__(self, memory_manager, config: Dict[str, Any]):
        """
        생명주기 관리자 초기화
        
        Args:
            memory_manager: 메모리 관리자
            config: 설정 딕셔너리
        """
        self.memory_manager = memory_manager
        self.config = config
        
        # 설정값
        self.decay_factor = config['network'].get('decay_factor', 0.95)
        self.promotion_threshold = config['network'].get('promotion_threshold', 0.8)
        self.cleanup_interval_hours = 24  # 기본 24시간마다 정리
        
        # 마지막 작업 시간
        self.last_decay_time = datetime.now()
        self.last_promotion_time = datetime.now()
        self.last_cleanup_time = datetime.now()
        
        # 통계
        self.decay_count = 0
        self.promotion_count = 0
        self.cleanup_count = 0
    
    async def run_lifecycle_cycle(self) -> Dict[str, Any]:
        """
        생명주기 관리 작업 실행
        
        Returns:
            작업 결과 통계
        """
        stats = {}
        
        # 1. 연결 강도 감쇠 적용
        decay_result = await self.apply_decay()
        stats['decay'] = decay_result
        
        # 2. 메모리 승격 작업
        promotion_result = await self.promote_memories()
        stats['promotion'] = promotion_result
        
        # 3. 정리 작업
        cleanup_result = await self.cleanup_old_memories()
        stats['cleanup'] = cleanup_result
        
        return stats
    
    async def apply_decay(self) -> Dict[str, Any]:
        """
        연결 강도 감쇠 적용
        
        Returns:
            감쇠 작업 결과
        """
        # Neo4j 저장소 감쇠
        await self.memory_manager.apply_decay(self.decay_factor)
        
        # 마지막 감쇠 시간 업데이트
        self.last_decay_time = datetime.now()
        self.decay_count += 1
        
        return {
            'applied_at': self.last_decay_time.isoformat(),
            'decay_factor': self.decay_factor,
            'count': self.decay_count
        }
    
    async def promote_memories(self) -> Dict[str, Any]:
        """
        메모리 승격 작업
        
        Returns:
            승격 작업 결과
        """
        # 약한 연결 기반 승격
        promoted_count = await self.memory_manager.promote_weak_connections_to_long_term(
            threshold=self.promotion_threshold
        )
        
        # 마지막 승격 시간 업데이트
        self.last_promotion_time = datetime.now()
        self.promotion_count += 1
        
        return {
            'applied_at': self.last_promotion_time.isoformat(),
            'threshold': self.promotion_threshold,
            'promoted_count': promoted_count,
            'count': self.promotion_count
        }
    
    async def cleanup_old_memories(self) -> Dict[str, Any]:
        """
        오래된 메모리 정리 작업
        
        Returns:
            정리 작업 결과
        """
        # (실제 구현은 Neo4j 저장소에 추가 필요)
        # 여기서는 가상의 결과만 반환
        
        # 마지막 정리 시간 업데이트
        self.last_cleanup_time = datetime.now()
        self.cleanup_count += 1
        
        return {
            'applied_at': self.last_cleanup_time.isoformat(),
            'interval_hours': self.cleanup_interval_hours,
            'cleaned_count': 0,  # 가상 값
            'count': self.cleanup_count
        }
