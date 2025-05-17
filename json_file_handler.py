"""#
JSON 파일에서 토픽/서브토픽 데이터를 로드하는 유틸리티
"""
import json
import os
from typing import List, Dict, Any

def load_json_topics_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    JSON 파일에서 토픽/서브토픽 데이터 로드
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        토픽/서브토픽 데이터 리스트
    """
    if not os.path.exists(file_path):
        print(f"경고: 파일을 찾을 수 없습니다: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return []
    except Exception as e:
        print(f"파일 로딩 오류: {e}")
        return []

def save_json_topics_to_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """
    토픽/서브토픽 데이터를 JSON 파일로 저장
    
    Args:
        data: 토픽/서브토픽 데이터 리스트
        file_path: 저장할 파일 경로
        
    Returns:
        성공 여부
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"파일 저장 오류: {e}")
        return False

# 샘플 JSON 데이터 생성 함수
def create_sample_json_file(file_path: str = "sample_topics.json") -> str:
    """
    샘플 JSON 데이터 파일 생성
    
    Args:
        file_path: 저장할 파일 경로
        
    Returns:
        생성된 파일 경로
    """
    # 샘플 데이터
    sample_data = [
        {
            "user_id": "user1",
            "sub_topic": "name",
            "topic": "basic_info",
            "memo": "조현호"
        },
        {
            "topic": "basic_info",
            "user_id": "user1",
            "sub_topic": "age",
            "memo": "29"
        },
        {
            "sub_topic": "foods",
            "topic": "interest",
            "user_id": "user1",
            "memo": "사용자는 치킨과 소고기를 좋아함"
        },
        {
            "user_id": "user1",
            "sub_topic": "previous_projects",
            "topic": "work",
            "memo": "어제(2025/05/03) 만든 프로그램 유지보수하느라 많은 시간 소모됨"
        },
        {
            "user_id": "user1",
            "sub_topic": "psychological_state",
            "topic": "work",
            "memo": "프로그램 문제로 인해 스트레스 느낌, \"젠장\" 표현에서 불만과 피로감 추정"
        },
        {
            "sub_topic": "feelings",
            "topic": "psychological",
            "user_id": "user1",
            "memo": "프로그램 문제가 생겨서 불만족감 표현 (\"젠장\")"
        },
        {
            "topic": "life_event",
            "user_id": "user1",
            "sub_topic": "appointment",
            "memo": "사용자 약속이 2025/05/11에 취소된 것으로 추정됨 (내일)"
        },
        {
            "sub_topic": "social_activities",
            "user_id": "user1",
            "topic": "psychological",
            "memo": "내일 예정된 약속이 취소된 것으로 보임"
        },
        {
            "sub_topic": "appointment_change",
            "topic": "life_event",
            "user_id": "user1",
            "memo": "내일 약속 취소 가능성 있음 (2025/05/08)"
        }
    ]
    
    # 파일 저장
    success = save_json_topics_to_file(sample_data, file_path)
    
    if success:
        print(f"샘플 JSON 파일이 생성되었습니다: {file_path}")
        return file_path
    else:
        print("샘플 JSON 파일 생성 실패")
        return ""
