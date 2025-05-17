
# 토픽과 서브토픽 연결 시스템 - Neo4j 기반 지식 그래프

## 프로젝트 개요 및 목적 🎯

이 프로젝트는 LLM(대규모 언어 모델) 기반 챗봇의 **장기 기억 기능을 강화**하기 위한 시스템이다. 대화에서 나온 토픽(상위개념)과 서브토픽(하위개념) 간의 연결 구조를 Neo4j 그래프 데이터베이스에 저장하고, **시간에 따른 기억 관리 메커니즘**을 구현하였다.

### 프로젝트의 핵심 목적

1. **대화 컨텍스트의 지속적 유지와 관리**
   일반 챗봇은 컨텍스트 윈도우 제한으로 과거 대화를 기억하지 못함. 본 시스템은 중요 개념을 그래프로 저장하여 장기간 대화 맥락 유지.

2. **연관 개념 간 지식 그래프 구축**
   대화에서 추출한 개념 간 관계를 자동으로 그래프화하고, 연결 강도를 통해 연관성을 정량화.

3. **인간의 기억 시스템 모방**
   워킹 메모리(초단기) → 단기 기억 → 장기 기억의 3단계 구조를 구현하고, 시간에 따른 연결 감쇠와 중요 정보 승격 메커니즘을 적용.

4. **대화 시스템의 연속성 및 개인화 강화**
   사용자별 지식 그래프를 통한 개인화된 응답 생성. 과거 대화에서 언급된 정보를 적절히 참조함.

## 실제 활용 사례 및 기대 효과 💡

### 개인 비서 AI 및 대화 시스템

수개월 전 언급한 정보도 자연스럽게 참조 가능.
예: "지난번에 말했던 여행 계획 기억나?" → 관련 토픽 전체 조회

### 지식 관리 시스템

대화를 통해 학습한 정보를 체계적인 지식 그래프로 구조화.
개념 간 연결을 시각적으로 탐색 가능

### 교육용 어플리케이션

학습자의 관심 주제와 이해도를 그래프로 표현하고, 맞춤형 학습 경로를 제시

### 전문 영역 상담 시스템

의료, 법률 등 복잡한 도메인 지식을 구조화하고, 이전 상담 내용을 기억해 적절히 연결

## 이 시스템으로 무엇을 할 수 있나요? 🚀

* 토픽과 서브토픽 간의 **연결 관계를 Neo4j에 자동 생성**
* 서로 다른 토픽 간의 **관련성을 공유 서브토픽 기반으로 계산**
* 개념 간 연결을 **시각적으로 표현**하여 정보 구조 파악
* 약한 연결을 가진 개념들을 **자동으로 장기 기억으로 이동**
* **초단기/단기/장기 기억 계층**별로 효율적 관리
* **그래프 DB + 벡터 DB** 조합으로 연산 효율과 탐색 능력 향상

## 기존 챗봇 시스템과의 차별점 🔍

| 기능       | 일반 챗봇  | 본 시스템         |
| -------- | ------ | ------------- |
| 장기 기억    | 제한적    | 그래프 기반 저장     |
| 개념 간 연결  | 없음     | 연결 강도 기반 구조화  |
| 기억 감쇠/승격 | 없음     | 시간 기반 알고리즘 존재 |
| 연관성 탐색   | 어려움    | 그래프 기반 탐색 가능  |
| 정보 표현    | 텍스트 위주 | 계층적 구조        |
| 시각화      | 미지원    | 그래프 시각화 지원    |

## Neo4j vs SQL: 언제 그래프 데이터베이스를 선택해야 할까요? 🤔

### Neo4j가 유리한 경우

* 복잡한 개념 간 **관계 중심 데이터**
* **경로 탐색**과 **패턴 매칭**이 필요한 경우
* 연속적 관계 탐색 (예: “이 개념과 간접적으로 연결된 것까지 보여줘”)

```cypher
MATCH (n:Person {name: "John"})-[*1..3]-(connected)
RETURN connected;
```

### SQL이 유리한 경우

* 명확한 **테이블 스키마**
* **집계/통계 연산**이 중심인 경우
* 대량의 레코드 처리
* **트랜잭션 중심** 시스템

실제 서비스에서는 **Neo4j + SQL + 벡터DB**의 하이브리드 조합이 가장 효율적이다.

## 기억 시스템 아키텍처 🧠

### 1. 워킹 메모리 (초단기 기억)

* 메모리 내 임시 저장
* 새로운 정보로 빠르게 대체됨

### 2. 단기 기억

* 최근 대화 정보 저장
* Neo4j에 개념 노드 및 관계로 구성됨

### 3. 장기 기억

* 중요 정보 또는 자주 참조되는 정보 저장
* ChromaDB 기반 벡터 저장소에 의미 기반 검색 가능

### 승격 및 감쇠 조건

* 중요도 ≥ 0.7, 접근 횟수 ≥ 3 → 장기 기억 승격
* 모든 연결은 주기적으로 0.95의 감쇠 계수 적용
  (최소 강도 0.1 이하면 삭제)

## 시작하기 🏁

### Neo4j 설치 및 실행 (Ubuntu)

```bash
curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neo4j-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/neo4j-archive-keyring.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

### 비밀번호 설정

```bash
cypher-shell -u neo4j -p neo4j
> ALTER USER neo4j SET PASSWORD 'your_new_password';
```

### Python 패키지 설치

```bash
pip install neo4j networkx matplotlib numpy asyncio chromadb sentence-transformers
```

### 실행 예시

```bash
python interactive_test_neo4j.py --uri bolt://localhost:7687 --user neo4j --password your_new_password
```

## 프로젝트 구조 📁

```
project/
├── enums.py
├── memory_entry.py
├── neo4j_config.py
├── neo4j_storage.py
├── memory_manager_neo4j.py
├── neo4j_associative_chatbot.py
├── session.py
├── visualization.py
├── json_file_handler.py
├── utils.py
├── vector_storage.py
├── interactive_test_neo4j.py
└── README.md
```

## 연결 강도 산정 방식 💪

* 토픽 → 서브토픽: 0.9 (`CONTAINS`)
* 서브토픽 ↔ 서브토픽: 0.7 (`RELATED`)
* 토픽 ↔ 토픽: `0.5 + (공유 서브토픽 수 × 0.1)`, 최대 0.9
* 서브토픽 → 키워드: 0.6 (`RELATES_TO`)

## JSON 데이터 포맷 예시 📊

```json
[
  {
    "user_id": "user1",
    "topic": "basic_info",
    "sub_topic": "name",
    "memo": "조현호"
  }
]
```

## Neo4j 브라우저에서 연결 강도 확인하기 🔍

```cypher
MATCH (a:Concept)-[r]->(b:Concept)
WHERE r.weight IS NOT NULL
RETURN a.name AS Source, b.name AS Target, r.weight AS Strength
ORDER BY Strength DESC;
```

## 그래프 시각화 및 탐색 가이드 🎨

Neo4j 브라우저(`http://localhost:7474`)에서는 아래와 같은 쿼리로 지식 그래프를 직관적으로 탐색할 수 있다:

```cypher
// 전체 개념 노드 조회
MATCH (c:Concept) RETURN c LIMIT 25;

// 토픽과 서브토픽 관계
MATCH (t:Concept {type: 'topic'})-[r]->(s:Concept {type: 'subtopic'}) RETURN t, r, s;

// 특정 토픽 중심 탐색
MATCH path = (t:Concept {name: 'basic_info'})-[*1..2]-(other) RETURN path;

// 토픽 간 의미적 관계
MATCH (t1:Concept {type: 'topic'})-[r]-(t2:Concept {type: 'topic'}) RETURN t1, r, t2;

// 약한 연결 탐색
MATCH (a:Concept)-[r]->(b:Concept) WHERE r.weight <= 0.5 RETURN a, r, b;

// 감쇠 시뮬레이션
MATCH (a:Concept)-[r]->(b:Concept)
WHERE r.weight IS NOT NULL
WITH a.name AS Source, b.name AS Target, r.weight AS CurrentStrength,
     r.weight * 0.95 AS After1Decay,
     r.weight * 0.95 * 0.95 AS After2Decays
RETURN Source, Target, CurrentStrength, After1Decay, After2Decays;
```

### 팁

* 노드 클릭 시 관련 연결 강조
* 마우스 휠로 확대/축소
* 우측 하단에서 PNG, SVG 형식으로 그래프 저장 가능

---

## 향후 발전 방향 🔮

1. **자동 토픽 추출**: LLM 기반 자동 개념 추출
2. **다중 DB 연동**: Neo4j + SQL + 벡터DB 조합
3. **사용자 피드백 기반 연결 강도 조정**
4. **시간/공간 기반 지식 그래프 확장**
5. **웹 기반 대시보드**로 실시간 탐색 지원

---

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

---

