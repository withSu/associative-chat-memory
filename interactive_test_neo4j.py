"""#
Neo4j 기반 대화형 테스트 스크립트
Neo4j를 사용하여 토픽/서브토픽 관계 설정 및 시각화
"""
import asyncio
import json
from datetime import datetime
import os
import argparse
# 수정: json_file_handler에서 필요한 함수 임포트
from json_file_handler import load_json_topics_from_file, create_sample_json_file
from neo4j_config import Neo4jSystemConfig
from neo4j_associative_chatbot import Neo4jAssociativeChatbot
from enums import ConnectionType  # 경로 수정

async def run_neo4j_test(json_file_path: str = None, config: Neo4jSystemConfig = None):
    """
    Neo4j 기반 대화형 JSON 처리 및 시각화 테스트
    
    Args:
        json_file_path: JSON 파일 경로 (없으면 샘플 생성)
        config: Neo4j 설정 (없으면 기본값 사용)
    """
    # 1. JSON 파일 확인 및 로드
    if not json_file_path or not os.path.exists(json_file_path):
        print("JSON 파일이 지정되지 않았거나 찾을 수 없습니다. 샘플 파일을 생성합니다.")
        json_file_path = create_sample_json_file()
    
    json_data = load_json_topics_from_file(json_file_path)
    if not json_data:
        print("JSON 데이터를 로드할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    # 2. 시스템 초기화
    if config is None:
        config = Neo4jSystemConfig()
    
    chatbot = Neo4jAssociativeChatbot(config)
    
    print(f"=== Neo4j 기반 대화형 JSON 처리 테스트 ===")
    print(f"JSON 파일: {json_file_path}")
    print(f"로드된 데이터: {len(json_data)}개 항목")
    print(f"Neo4j 연결 URI: {config.neo4j_uri}")
    
    # 토픽 목록 추출
    topics = set(item["topic"] for item in json_data)
    print(f"토픽 목록: {topics}")
    
    # 그래프 구축 완료 여부
    graph_built = False
    
    # 3. 대화형 메뉴
    while True:
        print("\n==== 메뉴 ====")
        print("1. Neo4j에 토픽/서브토픽 관계 설정")
        print("2. 모든 연결 시각화")
        print("3. 토픽 간 연결 시각화")
        print("4. 특정 토픽 내부 연결 시각화")
        print("5. 약한 연결 찾기 및 메모리 승격")
        print("6. Neo4j 데이터베이스 통계 보기")
        print("7. JSON 데이터 보기")
        print("8. Neo4j 연결 테스트")
        print("0. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            # Neo4j에 토픽/서브토픽 관계 설정
            print("\n토픽/서브토픽 관계를 Neo4j에 설정 중...")
            await chatbot.memory_manager.save_topic_subtopic_relations(json_data)
            graph_built = True
            print("관계 설정 완료!")
        elif choice == "2":
            # 모든 연결 시각화
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1을 선택하세요.")
                continue
            
            print("\n모든 연결 시각화...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"neo4j_all_connections_{timestamp}.png"
            
            # 전체 그래프 시각화
            result = await chatbot.visualize_all_connections(viz_path)
            if result:
                print(f"모든 연결 네트워크가 저장되었습니다: {viz_path}")
                # 이미지 파일 열기
                try:
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(viz_path)
                    elif platform.system() == 'Darwin':  # macOS
                        import subprocess
                        subprocess.run(['open', viz_path])
                    else:  # Linux
                        import subprocess
                        subprocess.run(['xdg-open', viz_path])
                    print("이미지 파일을 열었습니다.")
                except Exception as e:
                    print(f"이미지 파일을 자동으로 열 수 없습니다: {e}")
            else:
                print("시각화 실패! 그래프에 노드가 없거나 문제가 있습니다.")
        elif choice == "3":
            # 토픽 간 연결 시각화
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1을 선택하세요.")
                continue
            
            print("\n토픽 간 연결 시각화...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"neo4j_topics_network_{timestamp}.png"
            
            # 토픽 간 연결 시각화 함수 호출
            result = await chatbot.visualize_topics_network(viz_path)
            if result:
                print(f"토픽 간 연결 네트워크가 저장되었습니다: {viz_path}")
                # 이미지 파일 열기
                try:
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(viz_path)
                    elif platform.system() == 'Darwin':  # macOS
                        import subprocess
                        subprocess.run(['open', viz_path])
                    else:  # Linux
                        import subprocess
                        subprocess.run(['xdg-open', viz_path])
                    print("이미지 파일을 열었습니다.")
                except Exception as e:
                    print(f"이미지 파일을 자동으로 열 수 없습니다: {e}")
            else:
                print("시각화 실패! 그래프에 토픽이 없거나 문제가 있습니다.")
        elif choice == "4":
            # 특정 토픽 내부 연결 시각화
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1을 선택하세요.")
                continue
                
            print("\n특정 토픽 내부 연결 시각화...")
            print(f"토픽 목록: {topics}")
            topic = input("토픽을 선택하세요: ").strip().lower()
            
            if not topic:
                print("유효한 토픽을 선택해주세요.")
                continue
                
            # 대소문자 일치 찾기
            found = False
            for t in topics:
                if t.lower() == topic:
                    topic = t  # 실제 대소문자 사용
                    found = True
                    break
                    
            if not found:
                print("유효한 토픽을 선택해주세요.")
                continue
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"neo4j_topic_internal_{topic}_{timestamp}.png"
            
            # 특정 토픽과 그 서브토픽 시각화
            result = await chatbot.visualize_topic_internal(topic, viz_path)
            if result:
                print(f"토픽 내부 연결 네트워크가 저장되었습니다: {viz_path}")
                # 이미지 파일 열기
                try:
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(viz_path)
                    elif platform.system() == 'Darwin':  # macOS
                        import subprocess
                        subprocess.run(['open', viz_path])
                    else:  # Linux
                        import subprocess
                        subprocess.run(['xdg-open', viz_path])
                    print("이미지 파일을 열었습니다.")
                except Exception as e:
                    print(f"이미지 파일을 자동으로 열 수 없습니다: {e}")
            else:
                print("시각화 실패! 그래프에 해당 토픽이 없거나 문제가 있습니다.")
        elif choice == "5":
            # 약한 연결 찾기 및 메모리 승격
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1을 선택하세요.")
                continue
                
            print("\n약한 연결 찾기 및 메모리 승격...")
            threshold = float(input("연결 강도 임계값을 입력하세요 (0.1-0.9, 기본: 0.5): ") or "0.5")
            
            # 약한 연결 조회
            weak_connections = await chatbot.memory_manager.get_weak_connections(threshold)
            
            if not weak_connections:
                print(f"연결 강도 {threshold} 이하의 약한 연결이 없습니다.")
                continue
                
            print(f"\n연결 강도 {threshold} 이하의 약한 연결들:")
            for i, conn in enumerate(weak_connections[:10], 1):  # 최대 10개까지 출력
                print(f"{i}. {conn['source']} → {conn['target']} (강도: {conn['weight']:.2f}, 유형: {conn['type']})")
            
            if len(weak_connections) > 10:
                print(f"... 외 {len(weak_connections) - 10}개")
                
            promote = input("\n약한 연결 관련 메모리를 장기 기억으로 승격시키겠습니까? (y/n): ").lower() == 'y'
            
            if promote:
                # 메모리 승격
                promoted_count = await chatbot.memory_manager.promote_weak_connections_to_long_term(threshold)
                if promoted_count > 0:
                    print(f"{promoted_count}개의 메모리를 장기 기억으로 승격했습니다.")
                else:
                    print("승격할 메모리가 없습니다.")
        elif choice == "6":
            # Neo4j 데이터베이스 통계 보기
            stats = await chatbot.get_system_stats()
            print("\nNeo4j 데이터베이스 통계:")
            print(json.dumps(stats, indent=2))
        elif choice == "7":
            # JSON 데이터 보기
            print("\nJSON 데이터:")
            
            # 처음 5개 항목만 출력
            for i, item in enumerate(json_data[:5], 1):
                print(f"{i}. {json.dumps(item, ensure_ascii=False)}")
                
            if len(json_data) > 5:
                print(f"\n... 외 {len(json_data) - 5}개 항목")
            
            print(f"\n총 {len(json_data)}개 항목이 있습니다.")
        elif choice == "8":
            # Neo4j 연결 테스트
            print("\nNeo4j 연결 테스트...")
            try:
                # 간단한 쿼리 실행
                await asyncio.to_thread(
                    chatbot.memory_manager.neo4j._test_connection
                )
                print("Neo4j 연결 테스트 성공!")
            except Exception as e:
                print(f"Neo4j 연결 테스트 실패: {e}")
                
                # 연결 정보 출력
                print("\nNeo4j 연결 정보:")
                print(f"URI: {config.neo4j_uri}")
                print(f"사용자: {config.neo4j_user}")
                print("비밀번호: ********")
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")
    
    # 4. 종료
    await chatbot.shutdown()
    print("\n프로그램 종료")


if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="Neo4j 기반 대화형 JSON 처리 및 시각화 테스트")
    parser.add_argument('-f', '--file', help="JSON 파일 경로")
    parser.add_argument('--uri', help="Neo4j URI (예: neo4j://localhost:7687)", default="neo4j://localhost:7687")
    parser.add_argument('--user', help="Neo4j 사용자 이름", default="neo4j")
    parser.add_argument('--password', help="Neo4j 비밀번호", default="password")
    args = parser.parse_args()
    
    # Neo4j 설정
    config = Neo4jSystemConfig(
        neo4j_uri=args.uri,
        neo4j_user=args.user,
        neo4j_password=args.password
    )
    
    # 테스트 실행
    asyncio.run(run_neo4j_test(args.file, config))
    
    
    
    
    #python interactive_test_neo4j.py --uri bolt://localhost:7687 --user neo4j --password 0000