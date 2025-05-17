"""#
네트워크 시각화 모듈 - 연결 강도 기반 시각화 
추가 시각화 유틸리티 함수 제공
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import matplotlib.font_manager as fm
import os
import matplotlib.colors as mcolors

# 한글 폰트 설정 개선
def set_korean_font():
    """
    시스템에 설치된 한글 폰트를 찾아 설정
    """
    # 일반 폰트 설정
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 한글 폰트 목록
    korean_fonts = [
        'NanumGothic',
        'Malgun Gothic',
        'Apple SD Gothic Neo',
        'AppleGothic',
        'Noto Sans CJK KR',
        'Gulim',
        'Dotum',
        'Batang'
    ]
    
    # sans-serif에 한글 폰트 추가
    for font_name in korean_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and not font_path.endswith('DejaVuSans.ttf'):
                # sans-serif 패밀리에 한글 폰트 추가
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
                print(f"한글 폰트 설정: {font_name}")
                break
        except Exception as e:
            print(f"폰트 {font_name} 설정 실패: {e}")
            continue
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
    return plt.rcParams['font.sans-serif'][0] if plt.rcParams.get('font.sans-serif') else 'sans-serif'


def visualize_association_network(
    graph: nx.DiGraph,
    center_concept: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title_prefix: str = "연관 네트워크"
) -> Optional[str]:
    """
    연관 네트워크 시각화 - 연결 강도 기반 개선 버전
    
    Args:
        graph: NetworkX 그래프
        center_concept: 중심 개념
        save_path: 저장 경로
        show: 화면 표시 여부
        title_prefix: 제목 접두사
    
    Returns:
        저장된 파일 경로 또는 None
    """
    if center_concept and not graph.has_node(center_concept):
        print(f"경고: '{center_concept}' 개념이 그래프에 없습니다.")
        return None
        
    # 한글 폰트 설정
    try:
        font_name = set_korean_font()
    except Exception:
        font_name = 'sans-serif'

    # 피규어 설정 (크기 증가)
    plt.figure(figsize=(20, 18), dpi=100)  # 더 큰 크기로 설정
    
    # 여백 설정 - 제목과 범례를 위한 공간 확보
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)  # 상단 여백 더 크게
    
    # 서브그래프 생성 (중심 개념 기준)
    if center_concept and center_concept in graph:
        # 중심에서 2단계 깊이의 노드만 시각화
        subgraph_nodes = set([center_concept])
        subgraph_nodes.update(graph.neighbors(center_concept))
        
        # 두 번째 단계 이웃도 추가
        second_neighbors = set()
        for node in list(graph.neighbors(center_concept)):
            second_neighbors.update(list(graph.neighbors(node))[:8])  # 최대 8개 이웃
        subgraph_nodes.update(second_neighbors)
        
        # 서브그래프 생성
        subgraph = graph.subgraph(subgraph_nodes)
    else:
        subgraph = graph
    
    # 노드 속성 수집
    node_types = {}  # 토픽, 서브토픽 등 노드 유형 분류
    
    # 메타데이터에서 노드 타입 가져오기 (직접 구축 시 이미 설정됨)
    for node in subgraph.nodes():
        metadata = subgraph.nodes[node].get('metadata', {})
        if isinstance(metadata, dict) and 'type' in metadata:
            node_types[node] = metadata['type']
    
    # 메타데이터가 없는 경우 중심 개념 기반으로 유추
    if center_concept:
        if center_concept not in node_types:
            node_types[center_concept] = 'topic'
        for neighbor in graph.neighbors(center_concept):
            if neighbor not in node_types:
                # 연결 유형 확인
                edge_type = graph[center_concept][neighbor].get('type', 'semantic')
                # 계층적 관계면 서브토픽으로 판단
                if edge_type == 'hierarchical':
                    node_types[neighbor] = 'subtopic'
                else:
                    node_types[neighbor] = 'keyword'
            
            # 2단계 이웃은 키워드로 간주
            for second_neighbor in graph.neighbors(neighbor):
                if second_neighbor != center_concept and second_neighbor not in node_types:
                    node_types[second_neighbor] = 'keyword'
    
    # 노드 크기 계산 (활성화 횟수 및 노드 유형 기반)
    node_sizes = []
    for node in subgraph.nodes():
        activation_count = subgraph.nodes[node].get('activation_count', 0)
        
        # 노드 유형에 따른 기본 크기
        node_type = node_types.get(node, 'keyword')
        if node == center_concept:
            base_size = 2500  # 중심 노드
        elif node_type == 'topic':
            base_size = 2000  # 토픽
        elif node_type == 'subtopic':
            base_size = 1200  # 서브토픽
        else:
            base_size = 800   # 키워드
            
        # 활성화 횟수에 따른 보너스 크기
        size = base_size + min(activation_count * 100, 1000)
        node_sizes.append(size)
    
    # 노드 색상 설정 (개선)
    node_colors = []
    
    # 노드 유형별 색상
    type_colors = {
        'topic': '#ff3333',       # 빨간색 (토픽)
        'subtopic': '#3333ff',    # 파란색 (서브토픽)
        'keyword': '#33cc33'      # 녹색 (키워드)
    }
    
    for node in subgraph.nodes():
        if node == center_concept:
            node_colors.append('#aa0000')  # 중심 노드는 진한 빨간색
        else:
            # 노드 유형 기반 색상
            node_type = node_types.get(node, 'keyword')
            base_color = type_colors.get(node_type, '#87CEEB')
            
            # 접근 최근성에 따른 투명도 조정
            last_activated = subgraph.nodes[node].get('last_activated')
            if last_activated:
                time_ago = (datetime.now() - last_activated).total_seconds() / 3600
                fade = min(time_ago / 24, 0.7)  # 24시간 기준으로 페이드 (최대 0.7)
                # HTML 색상 코드를 RGB로 변환하고 알파 적용
                rgb = mcolors.to_rgb(base_color)
                node_colors.append((rgb[0], rgb[1], rgb[2], 1.0 - fade))
            else:
                node_colors.append(base_color)
    
    # 엣지 두께 설정 (연결 강도 기반)
    edge_widths = []
    edge_colors = []
    edge_styles = []
    
    # 연결 유형별 스타일
    connection_styles = {
        'hierarchical': 'solid',   # 계층적 관계: 실선
        'semantic': 'dashed',      # 의미적 관계: 점선
        'temporal': 'dashdot',     # 시간적 관계: 대시-점
        'emotional': 'dotted'      # 감정적 관계: 점선
    }
    
    for u, v in subgraph.edges():
        # 연결 강도 기반 두께
        weight = subgraph[u][v].get('weight', 0.5)
        # 비선형 매핑으로 연결 강도 시각적 차이 강화
        edge_widths.append(max(weight * 12, 1.0))  # 선 두께 증가
        
        # 연결 강도 기반 색상 (강한 연결=진한 색, 약한 연결=연한 색)
        if weight >= 0.7:
            # 강한 연결 (진한 색)
            edge_colors.append((0.1, 0.1, 0.1, min(weight * 1.2, 1.0)))
        elif weight >= 0.4:
            # 중간 연결 (중간 색)
            edge_colors.append((0.3, 0.3, 0.3, weight))
        else:
            # 약한 연결 (연한 색)
            edge_colors.append((0.5, 0.5, 0.5, max(weight, 0.3)))
        
        # 연결 유형 기반 스타일
        conn_type = subgraph[u][v].get('type', 'semantic')
        edge_styles.append(connection_styles.get(conn_type, 'solid'))
    
    # 레이아웃 계산 (중심 개념 고정)
    if center_concept and center_concept in subgraph:
        # 스프링 레이아웃 with 중심 고정
        pos = nx.spring_layout(
            subgraph,
            k=1.8/np.sqrt(len(subgraph.nodes())),  # 더 넓게 분포
            iterations=70,  # 반복 증가로 안정성 향상
            seed=42
        )
        # 중심 개념은 중앙에 고정
        pos[center_concept] = np.array([0, 0])
    else:
        # 일반 스프링 레이아웃
        pos = nx.spring_layout(
            subgraph,
            k=1.8/np.sqrt(len(subgraph.nodes())),
            iterations=70
        )
    
    # 연결 유형별로 그리기 (스타일 구분을 위해)
    edge_types = set()
    for _, _, data in subgraph.edges(data=True):
        edge_types.add(data.get('type', 'semantic'))
    
    for edge_type in edge_types:
        # 해당 유형의 엣지만 필터링
        edges_of_type = [(u, v) for u, v in subgraph.edges() if subgraph[u][v].get('type', 'semantic') == edge_type]
        
        if not edges_of_type:
            continue
            
        # 엣지 속성 필터링
        type_widths = [edge_widths[i] for i, (u, v) in enumerate(subgraph.edges()) if subgraph[u][v].get('type', 'semantic') == edge_type]
        type_colors = [edge_colors[i] for i, (u, v) in enumerate(subgraph.edges()) if subgraph[u][v].get('type', 'semantic') == edge_type]
        
        # 선 스타일 결정
        style = connection_styles.get(edge_type, 'solid')
        
        # 엣지 그리기
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=edges_of_type,
            width=type_widths,
            edge_color=type_colors,
            style=style,
            arrows=True,
            arrowsize=20,  # 화살표 크기 증가
            arrowstyle='-|>'
        )
    
    # 노드 그리기 (외곽선 추가로 구분 강화)
    for node_type in ['topic', 'subtopic', 'keyword']:
        # 해당 타입의 노드만 필터링
        nodes_of_type = [node for node in subgraph.nodes() if node_types.get(node) == node_type]
        if not nodes_of_type:
            continue
            
        # 타입별 사이즈 추출
        sizes = [node_sizes[i] for i, node in enumerate(subgraph.nodes()) if node_types.get(node) == node_type]
        
        # 타입별 색상 추출
        colors = [node_colors[i] for i, node in enumerate(subgraph.nodes()) if node_types.get(node) == node_type]
        
        # 외곽선 설정
        if node_type == 'topic':
            edgecolors = 'black'  # 토픽은 검정 외곽선
            linewidths = 3.0
        elif node_type == 'subtopic':
            edgecolors = 'darkblue'  # 서브토픽은 파란 외곽선
            linewidths = 2.0
        else:
            edgecolors = 'darkgreen'  # 키워드는 녹색 외곽선
            linewidths = 1.0
        
        # 각 타입별로 노드 그리기
        nx.draw_networkx_nodes(
            subgraph, pos,
            nodelist=nodes_of_type,
            node_size=sizes,
            node_color=colors,
            alpha=0.95,
            edgecolors=edgecolors,
            linewidths=linewidths
        )
    
    # 노드 라벨
    labels = {}
    for node in subgraph.nodes():
        # 토픽과 서브토픽에는 노드 타입 표시 추가
        node_type = node_types.get(node)
        
        if node_type == 'topic':
            labels[node] = f"[토픽] {node}"  # 토픽 라벨 특별 표시
        elif node_type == 'subtopic':
            labels[node] = f"[서브] {node}"  # 서브토픽 라벨 특별 표시
        else:
            labels[node] = str(node)
    
    # 토픽 라벨 먼저 그리기 (더 크게)
    topic_labels = {node: label for node, label in labels.items() if node_types.get(node) == 'topic'}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=topic_labels,
        font_size=16,  # 토픽은 더 큰 폰트
        font_weight='bold',
        font_family=font_name
    )
    
    # 서브토픽 라벨 그리기
    subtopic_labels = {node: label for node, label in labels.items() if node_types.get(node) == 'subtopic'}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=subtopic_labels,
        font_size=14,  # 서브토픽은 중간 크기 폰트
        font_weight='bold',
        font_family=font_name
    )
    
    # 키워드 라벨 그리기
    keyword_labels = {node: label for node, label in labels.items() if node_types.get(node) == 'keyword'}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=keyword_labels,
        font_size=12,  # 키워드는 작은 폰트
        font_family=font_name
    )
    
    # 엣지 가중치 표시 (강한 연결만)
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        weight = data.get('weight', 0)
        if weight > 0.6:  # 강한 연결만 표시
            edge_labels[(u, v)] = f"{weight:.2f}"
    
    nx.draw_networkx_edge_labels(
        subgraph, pos,
        edge_labels=edge_labels,
        font_size=11,
        font_family=font_name,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)  # 배경 추가로 가독성 향상
    )
    
    # 타이틀 설정 - 더 높은 위치에 배치
    title = f"{title_prefix}: {center_concept} 중심" if center_concept else f"{title_prefix}"
    plt.suptitle(title, fontsize=24, fontweight='bold', y=0.98)  # suptitle 사용하여 더 위에 배치
    
    # 범례 배경 설정 (가독성 향상)
    legend_bg = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='none')
    
    # 범례 위치 및 여백 설정 (더 위로 올림)
    # 노드 유형 범례 (왼쪽)
    plt.text(0.05, 0.95, "• 토픽 (빨간색)", color='#ff3333', fontweight='bold', fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    plt.text(0.05, 0.92, "• 서브토픽 (파란색)", color='#3333ff', fontweight='bold', fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    plt.text(0.05, 0.89, "• 키워드 (녹색)", color='#33cc33', fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    
    # 연결 강도 범례 (중앙) - 추가됨
    plt.text(0.35, 0.95, "강한 연결 (≥0.7): 굵은 선", fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    plt.text(0.35, 0.92, "중간 연결 (0.4~0.7): 중간 선", fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    plt.text(0.35, 0.89, "약한 연결 (<0.4): 얇은 선", fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    
    # 연결 유형 범례 (오른쪽)
    plt.text(0.65, 0.95, "계층적 관계: ——", fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    plt.text(0.65, 0.92, "의미적 관계: - - -", fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    plt.text(0.65, 0.89, "기타 관계: -·-", fontsize=12, 
             transform=plt.gcf().transFigure, bbox=legend_bg, ha='left')
    
    plt.axis('off')
    
    # 파일 저장 또는 화면 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def visualize_topics_network(
    graph: nx.DiGraph,
    topic_nodes: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title_prefix: str = "토픽 간 연결 네트워크"
) -> Optional[str]:
    """
    토픽 간 연결만 시각화하는 함수
    
    Args:
        graph: NetworkX 그래프
        topic_nodes: 토픽 노드 목록 (지정하지 않으면 그래프에서 추출)
        save_path: 저장 경로
        show: 화면 표시 여부
        title_prefix: 제목 접두사
        
    Returns:
        저장된 파일 경로 또는 None
    """
    # 토픽 노드 추출
    if not topic_nodes:
        topic_nodes = []
        for node, data in graph.nodes(data=True):
            # 메타데이터 확인
            metadata = data.get('metadata', {})
            if isinstance(metadata, dict) and metadata.get('type') == 'topic':
                topic_nodes.append(node)
    
    # 토픽이 없으면 실패
    if not topic_nodes:
        print("그래프에 토픽 노드가 없습니다.")
        return None
    
    # 토픽 노드만 포함하는 서브그래프 생성
    topics_subgraph = graph.subgraph(topic_nodes)
    
    # 서브그래프가 비어있으면 실패
    if topics_subgraph.number_of_nodes() == 0:
        print("토픽 노드가 연결되어 있지 않습니다.")
        return None
    
    # 시각화 함수 호출
    # center_concept은 None으로 지정하여 특정 중심 없이 전체 토픽 네트워크 표시
    return visualize_association_network(
        graph=topics_subgraph,
        center_concept=None,
        save_path=save_path,
        show=show,
        title_prefix=title_prefix
    )


def visualize_topic_internal(
    graph: nx.DiGraph,
    topic: str,
    save_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    특정 토픽과 그 서브토픽 및 관련 키워드만 시각화 (다른 토픽은 제외)
    
    Args:
        graph: NetworkX 그래프
        topic: 시각화할 토픽
        save_path: 저장 경로
        show: 화면 표시 여부
    
    Returns:
        저장된 파일 경로 또는 None
    """
    if not graph.has_node(topic):
        print(f"경고: '{topic}' 토픽이 그래프에 없습니다.")
        return None
    
    # 완전히 새로운 그래프 생성
    new_graph = nx.DiGraph()
    
    # 1. 선택한 토픽 노드 추가
    new_graph.add_node(topic)
    # 토픽 노드 속성 복사 또는 설정
    if 'metadata' in graph.nodes[topic]:
        new_graph.nodes[topic]['metadata'] = graph.nodes[topic]['metadata'].copy() if hasattr(graph.nodes[topic]['metadata'], 'copy') else graph.nodes[topic]['metadata']
    else:
        new_graph.nodes[topic]['metadata'] = {'type': 'topic'}
    
    # 2. 서브토픽 노드 추가 (계층적 관계를 가진 이웃들)
    subtopics = []
    for neighbor in graph.neighbors(topic):
        if topic in graph and neighbor in graph[topic]:  # edge 존재 확인
            edge_data = graph[topic][neighbor]
            # 계층적 관계만 서브토픽으로 처리
            if edge_data.get('type', '') == 'hierarchical':
                subtopics.append(neighbor)
                # 서브토픽 노드 추가
                new_graph.add_node(neighbor)
                # 속성 복사 또는 설정
                if 'metadata' in graph.nodes[neighbor]:
                    new_graph.nodes[neighbor]['metadata'] = graph.nodes[neighbor]['metadata'].copy() if hasattr(graph.nodes[neighbor]['metadata'], 'copy') else graph.nodes[neighbor]['metadata']
                else:
                    # 메타데이터가 없으면 서브토픽 타입으로 설정
                    new_graph.nodes[neighbor]['metadata'] = {'type': 'subtopic'}
                # 토픽→서브토픽 엣지 추가
                new_graph.add_edge(topic, neighbor, **edge_data)
    
    # 3. 키워드 노드 추가 (서브토픽에 연결된 "keyword" 타입 노드)
    for subtopic in subtopics:
        for neighbor in graph.neighbors(subtopic):
            # 토픽이나 서브토픽이 아닌 노드만 키워드로 처리
            if neighbor != topic and neighbor not in subtopics:
                # 키워드 노드 추가
                new_graph.add_node(neighbor)
                # 속성 복사 또는 설정
                if 'metadata' in graph.nodes[neighbor]:
                    new_graph.nodes[neighbor]['metadata'] = graph.nodes[neighbor]['metadata'].copy() if hasattr(graph.nodes[neighbor]['metadata'], 'copy') else graph.nodes[neighbor]['metadata']
                else:
                    new_graph.nodes[neighbor]['metadata'] = {'type': 'keyword'}
                
                # 서브토픽→키워드 엣지 추가
                if subtopic in graph and neighbor in graph[subtopic]:
                    edge_data = graph[subtopic][neighbor]
                    new_graph.add_edge(subtopic, neighbor, **edge_data)
    
    # 4. 서브토픽 간 연결 추가
    for i, subtopic1 in enumerate(subtopics):
        for subtopic2 in subtopics[i+1:]:
            if subtopic1 in graph and subtopic2 in graph[subtopic1]:
                edge_data = graph[subtopic1][subtopic2]
                new_graph.add_edge(subtopic1, subtopic2, **edge_data)
            if subtopic2 in graph and subtopic1 in graph[subtopic2]:
                edge_data = graph[subtopic2][subtopic1]
                new_graph.add_edge(subtopic2, subtopic1, **edge_data)
    
    # 시각화
    return visualize_association_network(
        graph=new_graph,
        center_concept=topic,
        save_path=save_path,
        show=show,
        title_prefix=f"토픽 내부 연결: {topic}"
    )
