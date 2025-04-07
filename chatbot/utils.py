# chatbot/utils.py
from typing import List, Dict
from .mymodel.stock_crawling import latest_news # 실제 서버로 돌릴때
# from mymodel.stock_crawling import latest_news # 테스트 할때
from symspellpy import SymSpell, Verbosity 
import json
import os
from dotenv import load_dotenv
import pandas as pd


# .env 불러오기
load_dotenv()

def extract_stock_name(msg):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "mymodel", "data", "stock_list.json")
    DATA_PATH = os.path.abspath(DATA_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        stock_data = json.load(f)

    stock_names = [item["회사명"] for item in stock_data]

    ALIAS_PATH = os.path.join(BASE_DIR, "mymodel", "data", "alias.csv")

    # CSV 불러오기
    alias_df = pd.read_csv(ALIAS_PATH, encoding="utf-8-sig")

    # alias → 정식 종목명 딕셔너리로 바로 변환
    alias_map = dict(zip(alias_df["alias"], alias_df["정식명"]))

    # 0️⃣ 먼저 alias로 바로 매핑 시도
    msg_no_space = msg.replace(" ", "")
    for alias, true_name in alias_map.items():
        if alias in msg_no_space:
            return true_name, None

    # 1️⃣ 정확한 종목명이 포함돼 있으면 바로 반환
    for name in stock_names:
        if name in msg:
            return name, None  # 두 번째는 선택지 없음

    # 2️⃣ SymSpell 초기화
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=5)
    for name in stock_names:
        sym_spell.create_dictionary_entry(name, 1)

    # 3️⃣ 공백 제거하고 오타 교정
    msg_no_space = msg.replace(" ", "")

    try:
        suggestions = sym_spell.lookup(msg_no_space, Verbosity.CLOSEST, max_edit_distance=2)
    except ValueError:
        return None, None

    # 4️⃣ 유사 종목 2개 제안
    suggestion_names = [s.term for s in suggestions[:2] if s.term in stock_names]

    if suggestion_names:
        return None, suggestion_names  # 정답은 없고, 선택지만 반환

    return None, None  # 아무것도 못 찾았을 때


def get_latest_news(stock_name: str) -> List[str]:
    """
    종목명을 받아서 최신 뉴스 본문 리스트를 반환
    예시에서는 더미 데이터, 실제론 API 요청 필요
    여기서는 상승 또는 하락만 따질것이므로 0,2가 하락, 1,3이 상승이다.
    이것만 판단되면 바로 return
    """

    news_list_data = latest_news(stock_name) # 뉴스 리스트 받기
    return news_list_data

def get_financial_risks(stock_name):
    """
    종목명 기반 재무 리스크 점수를 정규화해서 반환 (0~1)
    실제론 API + 정규화 로직 필요
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "mymodel", "data", "stock_list.json")
    DATA_PATH = os.path.abspath(DATA_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        stock_data = json.load(f)

    stock_code = ""

    for stock in stock_data:
        if stock["회사명"] == stock_name:
            stock_code = stock["종목코드"]  # {'stock_name': ..., 'stock_code': ..., 'market': ...}
            break

    ## 여기에 리스트 분석 함수 불러와서 return만 해줌

    return {
        "volatility": 0.6,
        "liquidity": 0.3,
        "financial": 0.4,
        "investor": 0.5
    }

def calculate_final_score(avg_probs, risk_data):
    """
    avg_probs: [4개 확률] 클래스 0~3 softmax 평균
    risk_data: dict (각 리스크 점수: 0~1)

    반환: 0~1 사이 점수 (높을수록 상승 가능성 높음)
    """
    # 뉴스 점수 (클래스 1,3 → 상승 관련)
    news_score = avg_probs[1] * 0.5 + avg_probs[3] * 0.75  # 소폭 상승 + 강력 상승

    # 리스크 점수 (리스크가 낮을수록 상승 가능성 ↑)
    risk_score = 1 - (
        0.25 * risk_data["volatility"] + # 변동성 리스크
        0.25 * risk_data["liquidity"] + # 	유동성 리스크
        0.25 * risk_data["financial"] + # 재무 리스크
        0.25 * risk_data["investor"] # 투자자 동향 리스크
    )

    final_score = (news_score * 0.5) + (risk_score * 0.5)
    return round(final_score, 4)

def interpret_score(score):
    if score >= 0.8:
        return "📈 매우 상승 가능성이 높아요"
    elif score >= 0.65:
        return "⬆ 상승 가능성이 있어 보여요"
    elif score >= 0.45:
        return "➖ 보합세일 수 있어요"
    elif score >= 0.3:
        return "⬇ 하락 위험이 있어 보여요"
    else:
        return "📉 크게 하락할 가능성이 있어요"

def generate_news_summary(news_list, predicted_class):
    """
    예시: 뉴스 수 + 예측 클래스 기반 간단 요약 출력
    """
    class_desc = {
        0: "📉 부정적인 뉴스가 많아 하락 가능성이 있습니다.",
        1: "⬆ 약간의 긍정 뉴스가 있어요.",
        2: "⬇ 약간의 부정 뉴스가 있습니다.",
        3: "📈 긍정적인 뉴스가 다수 포착됐어요.",
        4: "❔ 판단이 애매한 뉴스가 많아요."
    }
    return f"{len(news_list)}건의 뉴스 기반 예측: {class_desc.get(predicted_class, '정보 없음')}"

if __name__ == "__main__":
    print(extract_stock_name("밍동 항공"))