"""#
Neo4j 그래프 데이터베이스 저장소 클래스
토픽, 서브토픽, 키워드 간 연결 관계 관리
"""
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable

from memory_entry import MemoryEntry  # 또는 .models.memory_entry
from enums import MemoryTier, ConnectionType  # 또는 .models.enums


class Neo4jStorage:
    """
    Neo4j 저장소 - 연관 관계 중심 저장소
    
    특징:
    - 그래프 기반의 토픽/서브토픽 연결 관리
    - 연결 강도에 따른 자동 장기 기억 이관
    - 실시간 토폴로지 쿼리 및 시각화
    """
    
    def __init__(self, uri: str, user: str, password: str, 
                 min_connection_strength: float = 0.1, 
                 connection_strength_threshold: float = 0.5):
        """
        Neo4j 저장소 초기화
        
        Args:
            uri: Neo4j 데이터베이스 URI (예: "neo4j://localhost:7687")
            user: 데이터베이스 사용자 이름
            password: 데이터베이스 비밀번호
            min_connection_strength: 최소 연결 강도 (이 값 이하의 연결은 제거)
            connection_strength_threshold: 약한 연결로 판단하는 임계값
        """
        # Neo4j 연결 관리
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        
        # 설정값
        self.min_connection_strength = min_connection_strength
        self.connection_strength_threshold = connection_strength_threshold
        
        # 초기화
        self._initialize_constraints()
        
        # 통계
        self.stats = {
            'saved_memories': 0,
            'saved_connections': 0,
            'weak_connections': 0
        }
        
    def _initialize_constraints(self) -> None:
        """
        Neo4j 제약 조건 초기화 (노드 유일성 등)
        """
        with self.driver.session() as session:
            # 사용자 ID 고유성 제약
            session.execute_write(self._create_user_constraint)
            
            # 메모리 ID 고유성 제약
            session.execute_write(self._create_memory_constraint)
            
            # 개념(토픽, 서브토픽) 유일성 제약
            session.execute_write(self._create_concept_constraint)
    
    @staticmethod
    def _create_user_constraint(tx):
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
        
    @staticmethod
    def _create_memory_constraint(tx):
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE")
        
    @staticmethod
    def _create_concept_constraint(tx):
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE (c.name, c.type) IS UNIQUE")
    
    def close(self) -> None:
        """
        Neo4j 연결 종료
        """
        self.driver.close()
    
    async def save_memory(self, 
                          memory: MemoryEntry, 
                          relationships: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        메모리 저장 및 관계 설정
        
        Args:
            memory: 저장할 메모리 엔트리
            relationships: 관계 정보 목록 (없으면 자동 생성)
            
        Returns:
            저장된 메모리 ID
        """
        try:
            # Neo4j는 비동기를 네이티브로 지원하지 않으므로 ThreadPoolExecutor를 사용
            # asyncio.to_thread를 사용하여 비동기로 실행
            memory_id = await asyncio.to_thread(
                self._save_memory_sync, memory, relationships
            )
            
            self.stats['saved_memories'] += 1
            return memory_id
        except Exception as e:
            print(f"Neo4j 메모리 저장 오류: {e}")
            raise
    
    def _save_memory_sync(self, 
                          memory: MemoryEntry, 
                          relationships: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        메모리 저장 및 관계 설정 (동기 버전)
        """
        with self.driver.session() as session:
            # 메모리 저장
            session.execute_write(self._create_memory_node, memory)
            
            # 사용자-메모리 연결
            user_id = memory.content.get('user_id')
            if user_id:
                session.execute_write(self._create_user_memory_relationship, user_id, memory.id)
            
            # 관계가 명시적으로 제공되지 않았으면 메모리 내용에서 추출
            if not relationships and 'topic' in memory.content and 'sub_topic' in memory.content:
                relationships = [{
                    'source': memory.content['topic'],
                    'source_type': 'topic',
                    'target': memory.content['sub_topic'],
                    'target_type': 'subtopic',
                    'type': 'CONTAINS',
                    'properties': {
                        'weight': 0.9,  # 토픽-서브토픽 강한 연결
                        'connection_type': 'hierarchical'
                    }
                }]
            
            # 관계 생성 (제공된 경우)
            if relationships:
                for rel in relationships:
                    session.execute_write(
                        self._create_concept_relationship,
                        rel['source'], rel['source_type'],
                        rel['target'], rel['target_type'],
                        rel['type'], rel.get('properties', {})
                    )
                    
                    # 메모리에 토픽 및 서브토픽 연결
                    if rel['source_type'] == 'topic':
                        session.execute_write(
                            self._create_memory_concept_relationship,
                            memory.id, rel['source'], 'ABOUT_TOPIC'
                        )
                    if rel['target_type'] == 'subtopic':
                        session.execute_write(
                            self._create_memory_concept_relationship,
                            memory.id, rel['target'], 'ABOUT_SUBTOPIC'
                        )
            
            # 개념 생성 및 연결
            for concept in memory.concepts:
                session.execute_write(
                    self._create_concept_node, concept, 'keyword'
                )
                
                session.execute_write(
                    self._create_memory_concept_relationship,
                    memory.id, concept, 'HAS_KEYWORD'
                )
            
            return memory.id
    
    @staticmethod
    def _create_memory_node(tx, memory: MemoryEntry) -> None:
        """
        메모리 노드 생성 트랜잭션
        """
        # 메모리 내용 및 메타데이터를 JSON으로 직렬화
        content_json = json.dumps(memory.content)
        metadata_json = json.dumps(memory.metadata)
        concepts_json = json.dumps(memory.concepts)
        
        # 메모리 노드 생성 쿼리
        query = """
        MERGE (m:Memory {id: $id})
        ON CREATE SET
            m.content = $content,
            m.concepts = $concepts,
            m.importance = $importance,
            m.emotional_weight = $emotional_weight,
            m.access_count = $access_count,
            m.tier = $tier,
            m.metadata = $metadata,
            m.last_accessed = $last_accessed,
            m.creation_time = $creation_time
        ON MATCH SET
            m.content = $content,
            m.concepts = $concepts,
            m.importance = $importance,
            m.emotional_weight = $emotional_weight,
            m.access_count = $access_count,
            m.tier = $tier,
            m.metadata = $metadata,
            m.last_accessed = $last_accessed
        RETURN m
        """
        
        tx.run(query,
               id=memory.id,
               content=content_json,
               concepts=concepts_json,
               importance=memory.importance,
               emotional_weight=memory.emotional_weight,
               access_count=memory.access_count,
               tier=memory.tier.value,
               metadata=metadata_json,
               last_accessed=memory.last_accessed.isoformat() if memory.last_accessed else None,
               creation_time=memory.creation_time.isoformat()
        )
    
    @staticmethod
    def _create_user_memory_relationship(tx, user_id: str, memory_id: str) -> None:
        """
        사용자-메모리 관계 생성 트랜잭션
        """
        query = """
        MERGE (u:User {user_id: $user_id})
        WITH u
        MATCH (m:Memory {id: $memory_id})
        MERGE (u)-[r:REMEMBERS]->(m)
        ON CREATE SET r.created_at = $timestamp
        RETURN u, m
        """
        
        tx.run(query,
               user_id=user_id,
               memory_id=memory_id,
               timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def _create_concept_node(tx, name: str, concept_type: str) -> None:
        """
        개념 노드 생성 트랜잭션
        """
        query = """
        MERGE (c:Concept {name: $name, type: $type})
        ON CREATE SET
            c.created_at = $timestamp,
            c.activation_count = 1
        ON MATCH SET
            c.activation_count = c.activation_count + 1,
            c.last_activated = $timestamp
        RETURN c
        """
        
        tx.run(query,
               name=name,
               type=concept_type,
               timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def _create_concept_relationship(tx, 
                                     source_name: str, source_type: str,
                                     target_name: str, target_type: str,
                                     rel_type: str, properties: Dict[str, Any]) -> None:
        """
        개념 간 관계 생성 트랜잭션
        """
        # 개념 노드 생성 (존재하지 않는 경우)
        tx.run("""
        MERGE (s:Concept {name: $source_name, type: $source_type})
        ON CREATE SET s.created_at = $timestamp
        """, source_name=source_name, source_type=source_type, timestamp=datetime.now().isoformat())
        
        tx.run("""
        MERGE (t:Concept {name: $target_name, type: $target_type})
        ON CREATE SET t.created_at = $timestamp
        """, target_name=target_name, target_type=target_type, timestamp=datetime.now().isoformat())
        
        # 속성 준비
        props = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'strengthening_count': 1
        }
        props.update(properties)
        
        # 관계 생성 또는 업데이트
        query = f"""
        MATCH (s:Concept {{name: $source_name, type: $source_type}})
        MATCH (t:Concept {{name: $target_name, type: $target_type}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET
            r += $props
        ON MATCH SET
            r.weight = CASE
                WHEN r.weight IS NULL THEN $weight
                ELSE CASE
                    WHEN $weight > r.weight THEN $weight
                    ELSE r.weight
                END
            END,
            r.strengthening_count = COALESCE(r.strengthening_count, 0) + 1,
            r.updated_at = $timestamp
        RETURN s, r, t
        """
        
        tx.run(query,
               source_name=source_name,
               source_type=source_type,
               target_name=target_name,
               target_type=target_type,
               props=props,
               weight=properties.get('weight', 0.5),
               timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def _create_memory_concept_relationship(tx, memory_id: str, concept_name: str, rel_type: str) -> None:
        """
        메모리-개념 관계 생성 트랜잭션
        """
        query = f"""
        MATCH (m:Memory {{id: $memory_id}})
        MATCH (c:Concept {{name: $concept_name}})
        MERGE (m)-[r:{rel_type}]->(c)
        ON CREATE SET r.created_at = $timestamp
        RETURN m, r, c
        """
        
        tx.run(query,
               memory_id=memory_id,
               concept_name=concept_name,
               timestamp=datetime.now().isoformat()
        )
    
    async def process_json_keywords(self, json_data: List[Dict[str, Any]]) -> None:
        """
        JSON 형태로 전달받은 토픽과 서브토픽 정보 처리
        
        Args:
            json_data: 토픽/서브토픽 JSON 데이터
        """
        try:
            # Neo4j는 비동기를 네이티브로 지원하지 않으므로 ThreadPoolExecutor를 사용
            await asyncio.to_thread(self._process_json_keywords_sync, json_data)
        except Exception as e:
            print(f"Neo4j JSON 키워드 처리 오류: {e}")
            raise
    
    def _process_json_keywords_sync(self, json_data: List[Dict[str, Any]]) -> None:
        """
        JSON 키워드 처리 (동기 버전)
        """
        # 토픽-서브토픽 관계 추출 및 처리
        with self.driver.session() as session:
            # 1. 토픽 및 서브토픽 노드 생성
            for item in json_data:
                user_id = item.get('user_id')
                topic = item.get('topic')
                sub_topic = item.get('sub_topic')
                memo = item.get('memo', '')
                
                if not (user_id and topic and sub_topic):
                    continue
                
                # 사용자 노드 생성
                session.execute_write(self._create_user_node, user_id)
                
                # 토픽 노드 생성
                session.execute_write(self._create_concept_node, topic, 'topic')
                
                # 서브토픽 노드 생성
                session.execute_write(self._create_concept_node, sub_topic, 'subtopic')
                
                # 토픽-서브토픽 계층적 관계 생성
                session.execute_write(
                    self._create_concept_relationship,
                    topic, 'topic',
                    sub_topic, 'subtopic',
                    'CONTAINS',
                    {
                        'weight': 0.9,  # 높은 강도
                        'connection_type': 'hierarchical'  # 계층적 관계
                    }
                )
                
                # 메모에서 간단한 키워드 추출 (실제로는 더 정교한 방법 사용 가능)
                if memo:
                    keywords = [w for w in memo.split() if len(w) > 2]
                    for keyword in keywords[:3]:  # 최대 3개 키워드만 추출
                        # 키워드 노드 생성
                        session.execute_write(self._create_concept_node, keyword, 'keyword')
                        
                        # 서브토픽-키워드 연결 (의미적 관계)
                        session.execute_write(
                            self._create_concept_relationship,
                            sub_topic, 'subtopic',
                            keyword, 'keyword',
                            'RELATES_TO',
                            {
                                'weight': 0.6,  # 중간 강도
                                'connection_type': 'semantic'  # 의미적 관계
                            }
                        )
            
            # 2. 서브토픽 간 연결 생성 (같은 토픽에 속한 서브토픽끼리)
            topics = {}
            for item in json_data:
                topic = item.get('topic')
                sub_topic = item.get('sub_topic')
                
                if topic and sub_topic:
                    if topic not in topics:
                        topics[topic] = []
                    if sub_topic not in topics[topic]:
                        topics[topic].append(sub_topic)
            
            for topic, subtopics in topics.items():
                for i, subtopic1 in enumerate(subtopics):
                    for subtopic2 in subtopics[i+1:]:
                        # 서브토픽 간 관계 생성 (양방향)
                        session.execute_write(
                            self._create_concept_relationship,
                            subtopic1, 'subtopic',
                            subtopic2, 'subtopic',
                            'RELATED',
                            {
                                'weight': 0.7,  # 중간 강도
                                'connection_type': 'semantic'  # 의미적 관계
                            }
                        )
                        
                        session.execute_write(
                            self._create_concept_relationship,
                            subtopic2, 'subtopic',
                            subtopic1, 'subtopic',
                            'RELATED',
                            {
                                'weight': 0.7,  # 중간 강도
                                'connection_type': 'semantic'  # 의미적 관계
                            }
                        )
            
            # 3. 토픽 간 관계 생성
            topic_list = list(topics.keys())
            for i, topic1 in enumerate(topic_list):
                for topic2 in topic_list[i+1:]:
                    # 공유 서브토픽 수에 기반한 연결 강도
                    shared = set(topics[topic1]).intersection(set(topics[topic2]))
                    if shared:
                        # 공유 서브토픽이 많을수록 강한 연결
                        connection_strength = min(0.5 + (len(shared) * 0.1), 0.9)
                    else:
                        # 공유 서브토픽이 없으면 약한 연결
                        connection_strength = 0.4
                    
                    # 토픽 간 관계 생성 (양방향)
                    session.execute_write(
                        self._create_concept_relationship,
                        topic1, 'topic',
                        topic2, 'topic',
                        'RELATES_TO',
                        {
                            'weight': connection_strength,
                            'connection_type': 'semantic',
                            'shared_subtopics': len(shared) if shared else 0
                        }
                    )
                    
                    session.execute_write(
                        self._create_concept_relationship,
                        topic2, 'topic',
                        topic1, 'topic',
                        'RELATES_TO',
                        {
                            'weight': connection_strength,
                            'connection_type': 'semantic',
                            'shared_subtopics': len(shared) if shared else 0
                        }
                    )
    
    @staticmethod
    def _create_user_node(tx, user_id: str) -> None:
        """
        사용자 노드 생성 트랜잭션
        """
        query = """
        MERGE (u:User {user_id: $user_id})
        ON CREATE SET
            u.created_at = $timestamp
        RETURN u
        """
        
        tx.run(query,
               user_id=user_id,
               timestamp=datetime.now().isoformat()
        )
    
    async def get_weak_connections(self, threshold: float = None) -> List[Dict[str, Any]]:
        """
        지정된 임계값보다 약한 연결들을 조회
        
        Args:
            threshold: 연결 강도 임계값 (이 값 이하의 연결이 조회됨)
        
        Returns:
            약한 연결 목록
        """
        if threshold is None:
            threshold = self.connection_strength_threshold
            
        try:
            # 비동기적으로 약한 연결 조회
            return await asyncio.to_thread(self._get_weak_connections_sync, threshold)
        except Exception as e:
            print(f"Neo4j 약한 연결 조회 오류: {e}")
            return []
    
    def _get_weak_connections_sync(self, threshold: float) -> List[Dict[str, Any]]:
        """
        약한 연결 조회 (동기 버전)
        """
        with self.driver.session() as session:
            result = session.execute_read(self._read_weak_connections, threshold)
            
            # 결과 가공
            weak_connections = []
            for record in result:
                source = record['source']['name']
                source_type = record['source']['type']
                target = record['target']['name']
                target_type = record['target']['type']
                weight = record['relationship'].get('weight', 0.0)
                conn_type = record['relationship'].get('connection_type', 'semantic')
                
                weak_connections.append({
                    'source': source,
                    'source_type': source_type,
                    'target': target,
                    'target_type': target_type,
                    'weight': weight,
                    'type': conn_type
                })
            
            self.stats['weak_connections'] = len(weak_connections)
            return weak_connections
    
    @staticmethod
    def _read_weak_connections(tx, threshold: float):
        """
        약한 연결 조회 트랜잭션
        """
        query = """
        MATCH (source:Concept)-[r]->(target:Concept)
        WHERE r.weight <= $threshold
        RETURN source, r, target
        ORDER BY r.weight ASC
        """
        
        return list(tx.run(query, threshold=threshold))
    
    async def search_memories(self, 
                              concepts: List[str], 
                              query_text: Optional[str] = None, 
                              top_k: int = 10) -> List[Dict]:
        """
        메모리 검색
        
        Args:
            concepts: 검색할 개념 목록
            query_text: 검색 텍스트 (옵션)
            top_k: 반환할 최대 결과 수
            
        Returns:
            검색 결과 목록
        """
        try:
            # 비동기적으로 메모리 검색
            return await asyncio.to_thread(
                self._search_memories_sync, concepts, query_text, top_k
            )
        except Exception as e:
            print(f"Neo4j 메모리 검색 오류: {e}")
            return []
    
    def _search_memories_sync(self, 
                              concepts: List[str], 
                              query_text: Optional[str] = None, 
                              top_k: int = 10) -> List[Dict]:
        """
        메모리 검색 (동기 버전)
        """
        with self.driver.session() as session:
            result = session.execute_read(
                self._read_memories_by_concepts, concepts, top_k
            )
            
            # 결과 가공
            memories = []
            for record in result:
                memory_node = record['m']
                relevance = record['relevance']
                
                # 메모리 내용 파싱
                content = json.loads(memory_node['content']) if memory_node.get('content') else {}
                
                # MemoryEntry 생성
                memory = MemoryEntry(
                    id=memory_node['id'],
                    content=content,
                    concepts=json.loads(memory_node['concepts']) if memory_node.get('concepts') else [],
                    importance=memory_node.get('importance', 0.5),
                    emotional_weight=memory_node.get('emotional_weight', 0.0),
                    access_count=memory_node.get('access_count', 0),
                    tier=MemoryTier(memory_node.get('tier', 'short')),
                    metadata=json.loads(memory_node['metadata']) if memory_node.get('metadata') else {},
                    last_accessed=datetime.fromisoformat(memory_node['last_accessed']) if memory_node.get('last_accessed') else None,
                    creation_time=datetime.fromisoformat(memory_node['creation_time']) if memory_node.get('creation_time') else datetime.now()
                )
                
                memories.append({
                    'memory': memory,
                    'score': relevance,
                    'source': 'neo4j'
                })
            
            # 점수 기반 정렬
            memories.sort(key=lambda x: x['score'], reverse=True)
            
            return memories[:top_k]
    
    @staticmethod
    def _read_memories_by_concepts(tx, concepts: List[str], top_k: int):
        """
        개념으로 메모리 검색 트랜잭션
        """
        # 개념이 없으면 빈 리스트 반환
        if not concepts:
            return []
        
        # 개념 검색 쿼리 생성
        # 직접 관련된 메모리부터 찾은 후, 연관된 개념을 통한 메모리도 검색
        query = """
        MATCH (c:Concept)
        WHERE c.name IN $concepts
        
        // 직접 연결된 메모리 찾기
        OPTIONAL MATCH (m1:Memory)-[:HAS_KEYWORD|ABOUT_TOPIC|ABOUT_SUBTOPIC]->(c)
        
        // 연관된 개념을 통한 메모리 찾기
        OPTIONAL MATCH (c)-[r:RELATES_TO|CONTAINS|RELATED]-(related:Concept)<-[:HAS_KEYWORD|ABOUT_TOPIC|ABOUT_SUBTOPIC]-(m2:Memory)
        
        // 결과 병합
        WITH collect(distinct m1) + collect(distinct m2) AS all_memories
        UNWIND all_memories AS m
        
        // 중복 제거 및 관련성 점수 계산
        WITH m, 
             CASE
                WHEN m IS NOT NULL THEN
                    CASE 
                        WHEN any(x IN $concepts WHERE m.content CONTAINS x) THEN 1.0  // 컨텐츠에 개념 포함
                        ELSE 0.7  // 연관 개념을 통한 연결
                    END
                ELSE 0.0
             END AS relevance
        
        WHERE m IS NOT NULL AND relevance > 0
        RETURN m, relevance
        ORDER BY relevance DESC, m.importance DESC
        LIMIT $top_k
        """
        
        return list(tx.run(query, concepts=concepts, top_k=top_k))
    
    async def promote_weak_connections_to_long_term(self, threshold: float = None) -> int:
        """
        약한 연결 강도를 가진 개념들의 메모리를 장기 기억으로 승격
        
        Args:
            threshold: 약한 연결로 간주할 임계값 (없으면 기본값 사용)
            
        Returns:
            승격된 메모리 수
        """
        if threshold is None:
            threshold = self.connection_strength_threshold
            
        try:
            # 비동기적으로 약한 연결 기반 메모리 승격
            return await asyncio.to_thread(
                self._promote_weak_connections_sync, threshold
            )
        except Exception as e:
            print(f"Neo4j 약한 연결 기반 메모리 승격 오류: {e}")
            return 0
    
    def _promote_weak_connections_sync(self, threshold: float) -> int:
        """
        약한 연결 기반 메모리 승격 (동기 버전)
        """
        with self.driver.session() as session:
            # 1. 약한 연결 찾기
            weak_connections = self._get_weak_connections_sync(threshold)
            
            if not weak_connections:
                return 0
            
            # 약한 연결의 개념들 수집
            weak_concepts = set()
            for conn in weak_connections:
                weak_concepts.add(conn['source'])
                weak_concepts.add(conn['target'])
            
            # 2. 개념과 관련된 메모리 중에서 중요도 높은 것들 승격
            promoted_count = session.execute_write(
                self._promote_memories_for_concepts,
                list(weak_concepts)
            )
            
            return promoted_count
    
    @staticmethod
    def _promote_memories_for_concepts(tx, concepts: List[str]) -> int:
        """
        특정 개념과 관련된 메모리 중 승격 조건을 만족하는 것들을 장기 기억으로 승격
        """
        query = """
        MATCH (c:Concept)-[:HAS_KEYWORD|ABOUT_TOPIC|ABOUT_SUBTOPIC]-(m:Memory)
        WHERE c.name IN $concepts AND m.tier = 'short'
        AND (m.importance >= 0.7 OR m.access_count >= 3)
        SET m.tier = 'long'
        WITH count(m) AS promoted
        RETURN promoted
        """
        
        result = tx.run(query, concepts=concepts)
        return result.single()[0]
    
    async def visualize_graph(self, center_concept: Optional[str] = None) -> Dict[str, Any]:
        """
        Neo4j 그래프 시각화를 위한 데이터 추출
        
        Args:
            center_concept: 중심 개념 (없으면 전체 그래프)
            
        Returns:
            시각화 데이터 (NetworkX와 호환되는 형식)
        """
        try:
            # 비동기적으로 그래프 데이터 추출
            return await asyncio.to_thread(
                self._extract_graph_data_sync, center_concept
            )
        except Exception as e:
            print(f"Neo4j 그래프 데이터 추출 오류: {e}")
            return {'nodes': [], 'edges': []}
    
    def _extract_graph_data_sync(self, center_concept: Optional[str] = None) -> Dict[str, Any]:
        """
        그래프 데이터 추출 (동기 버전)
        """
        with self.driver.session() as session:
            if center_concept:
                # 중심 개념 기준 그래프 추출
                result = session.execute_read(
                    self._read_graph_with_center, center_concept
                )
            else:
                # 전체 개념 그래프 추출
                result = session.execute_read(self._read_full_graph)
            
            nodes = []
            edges = []
            
            # 노드 및 엣지 데이터 추출
            for record in result:
                # 노드 데이터
                for node_key in ['source', 'target']:
                    if record[node_key] and record[node_key]['name'] not in [n['name'] for n in nodes]:
                        node_data = {
                            'name': record[node_key]['name'],
                            'type': record[node_key]['type'],
                            'metadata': {
                                'type': record[node_key]['type'],
                                'creation_time': record[node_key].get('created_at', ''),
                                'activation_count': record[node_key].get('activation_count', 0)
                            }
                        }
                        nodes.append(node_data)
                
                # 엣지 데이터
                if record['relationship']:
                    edge_data = {
                        'source': record['source']['name'],
                        'target': record['target']['name'],
                        'weight': record['relationship'].get('weight', 0.5),
                        'type': record['relationship'].get('connection_type', 'semantic')
                    }
                    edges.append(edge_data)
            
            return {
                'nodes': nodes,
                'edges': edges
            }
    
    @staticmethod
    def _read_graph_with_center(tx, center_concept: str):
        """
        중심 개념 기준 그래프 조회 트랜잭션
        """
        query = """
        // 중심 개념 찾기
        MATCH (center:Concept)
        WHERE center.name = $center_concept
        
        // 직접 연결된 개념 (1단계)
        MATCH (center)-[r1]-(neighbor1:Concept)
        
        // 2단계까지 연결 확장 (제한적으로)
        OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2:Concept)
        WHERE neighbor2 <> center AND neighbor2.type = 'subtopic'
        LIMIT 50
        
        // 모든 관계 수집
        WITH collect({source: center, target: neighbor1, relationship: r1}) AS direct_relations,
             collect({source: neighbor1, target: neighbor2, relationship: r2}) AS indirect_relations
        
        // 결과 병합 및 반환
        UNWIND direct_relations + [rel IN indirect_relations WHERE rel.target IS NOT NULL] AS rel
        RETURN rel.source AS source, rel.target AS target, rel.relationship AS relationship
        """
        
        return list(tx.run(query, center_concept=center_concept))
    
    @staticmethod
    def _read_full_graph(tx):
        """
        전체 개념 그래프 조회 트랜잭션
        """
        query = """
        MATCH (s:Concept)-[r]->(t:Concept)
        RETURN s AS source, t AS target, r AS relationship
        LIMIT 500  // 너무 많은 데이터 방지
        """
        
        return list(tx.run(query))
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Neo4j 저장소 통계 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        try:
            # 비동기적으로 통계 데이터 추출
            db_stats = await asyncio.to_thread(self._get_stats_sync)
            
            # 내부 통계와 DB 통계 병합
            stats = {
                'neo4j': {
                    'total_memories': db_stats.get('memory_count', 0),
                    'total_concepts': db_stats.get('concept_count', 0),
                    'total_connections': db_stats.get('relationship_count', 0),
                    'memory_by_tier': db_stats.get('memory_by_tier', {}),
                    'concept_by_type': db_stats.get('concept_by_type', {})
                },
                'operations': self.stats
            }
            
            return stats
        except Exception as e:
            print(f"Neo4j 통계 조회 오류: {e}")
            return {'error': str(e)}
    
    def _get_stats_sync(self) -> Dict[str, Any]:
        """
        Neo4j 통계 조회 (동기 버전)
        """
        with self.driver.session() as session:
            stats = {}
            
            # 메모리 노드 수
            memory_count = session.execute_read(self._count_nodes, "Memory")
            stats['memory_count'] = memory_count
            
            # 개념 노드 수
            concept_count = session.execute_read(self._count_nodes, "Concept")
            stats['concept_count'] = concept_count
            
            # 관계 수
            relationship_count = session.execute_read(self._count_relationships)
            stats['relationship_count'] = relationship_count
            
            # 메모리 티어별 통계
            memory_by_tier = session.execute_read(self._count_by_property, "Memory", "tier")
            stats['memory_by_tier'] = memory_by_tier
            
            # 개념 타입별 통계
            concept_by_type = session.execute_read(self._count_by_property, "Concept", "type")
            stats['concept_by_type'] = concept_by_type
            
            return stats
    
    @staticmethod
    def _count_nodes(tx, label: str) -> int:
        """
        특정 레이블의 노드 수 카운트
        """
        query = f"MATCH (n:{label}) RETURN count(n) AS count"
        result = tx.run(query)
        return result.single()["count"]
    
    @staticmethod
    def _count_relationships(tx) -> int:
        """
        전체 관계 수 카운트
        """
        query = "MATCH ()-[r]->() RETURN count(r) AS count"
        result = tx.run(query)
        return result.single()["count"]
    
    @staticmethod
    def _count_by_property(tx, label: str, property_name: str) -> Dict[str, int]:
        """
        특정 속성 값별 노드 수 카운트
        """
        query = f"""
        MATCH (n:{label})
        RETURN n.{property_name} AS property_value, count(n) AS count
        """
        result = tx.run(query)
        
        counts = {}
        for record in result:
            value = record["property_value"] or "unknown"
            counts[value] = record["count"]
        
        return counts
    
    async def apply_decay(self, decay_factor: float = 0.95) -> None:
        """
        연결 강도에 감쇠 적용
        
        Args:
            decay_factor: 감쇠 계수 (0-1 사이 값)
        """
        try:
            # 비동기적으로 감쇠 적용
            await asyncio.to_thread(self._apply_decay_sync, decay_factor)
        except Exception as e:
            print(f"Neo4j 연결 강도 감쇠 적용 오류: {e}")
    
    def _apply_decay_sync(self, decay_factor: float) -> None:
        """
        연결 강도 감쇠 적용 (동기 버전)
        """
        with self.driver.session() as session:
            # 감쇠 적용
            session.execute_write(self._apply_decay_to_relationships, decay_factor, self.min_connection_strength)
    
    @staticmethod
    def _apply_decay_to_relationships(tx, decay_factor: float, min_strength: float) -> None:
        """
        관계 감쇠 적용 트랜잭션
        """
        # 모든 관계에 감쇠 적용
        query1 = """
        MATCH ()-[r]->()
        WHERE r.weight IS NOT NULL
        SET r.weight = r.weight * $decay_factor
        SET r.updated_at = $timestamp
        """
        
        # 약한 관계 삭제
        query2 = """
        MATCH ()-[r]->()
        WHERE r.weight < $min_strength
        DELETE r
        """
        
        tx.run(query1, decay_factor=decay_factor, timestamp=datetime.now().isoformat())
        tx.run(query2, min_strength=min_strength)
    
    def _test_connection(self) -> bool:
        """
        Neo4j 연결 테스트
        
        Returns:
            성공 여부
        """
        try:
            with self.driver.session() as session:
                # 간단한 쿼리 실행
                result = session.run("RETURN 1 AS test").single()
                if result and result["test"] == 1:
                    return True
                return False
        except Exception as e:
            print(f"Neo4j 연결 테스트 오류: {e}")
            return False