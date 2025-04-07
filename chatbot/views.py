from .mymodel.predict_stock_news_class import predict
from chatbot.utils import (get_latest_news,
                           get_financial_risks,calculate_final_score,
                           interpret_score,generate_news_summary,extract_stock_name)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os

@csrf_exempt
def chat(request):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "mymodel", "data", "stock_list.json")
    DATA_PATH = os.path.abspath(DATA_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        stock_data = json.load(f)

    stock_names = [item["회사명"] for item in stock_data]

    try:
        if request.method == "POST":
            body = json.loads(request.body)
            msg = body.get("message", "")

            session = request.session
            mode = session.get("mode", None)
            print("📦 session mode:", mode)

            if msg in ["초기화", "대화 초기화", "리셋"]:
                session["mode"] = None
                return JsonResponse({"response": ""})

            if mode == "usage_selection":
                session["mode"] = None
                if "1" in msg:
                    return JsonResponse({"response": (
                        "🤖 안녕하세요! Fivo bot이에요.\n"
                        "최신 뉴스를 분석해서 종목이 오를지, 떨어질지 예측해주는 똑똑한 AI가 있어요!\n"
                        "궁금한 종목 이름을 입력해보세요 😄"
                    )})
                elif "2" in msg:
                    return JsonResponse({"response": (
                        "🤖 주식 예측 사용법 안내\n\n"
                        "1. 예측을 원하는 💡주식 종목명💡을 입력하세요. 예: `삼성전자`, `카카오`, `네이버`\n"
                        "2. 입력한 종목과 관련된 💡최신 뉴스 데이터를 기반💡으로 주가 전망을 예측해드려요. 📈\n"
                        "3. 예측 결과는 5단계로 제공됩니다 : \n"
                        "- 📉 매우 하락\n"
                        "- ⬇ 하락\n"
                        "- ➖ 보합\n"
                        "- ⬆ 상승\n"
                        "- 📈 매우 상승\n\n"
                        "📝 예시: '삼성전자'"
                    )})
                elif "3" in msg:
                    return JsonResponse({"response": (
                        "🤖 트리맵 보는법 안내\n\n"
                        " - 트리맵은 테마별로 상승/하락 했는지 알 수 있습니다. \n"
                        " - 테마별 뉴스도 볼 수 있으니 참고해주세요! 📝\n"
                    )})
                elif "4" in msg:
                    return JsonResponse({"response": (
                        "🤖 캔들 차트 보는 법 안내\n\n"
                        " - 캔들 차트는 주가의 시가, 종가, 고가, 저가를 시각적으로 보여주는 차트입니다.\n"
                        " - 빨간색(양봉): 종가 > 시가 → 상승\n"
                        " - 파란색(음봉): 종가 < 시가 → 하락\n"
                        " - 차트의 꼬리(윗꼬리/아랫꼬리)는 변동 폭을 보여줍니다.\n"
                        " - 이를 통해 투자자 심리나 시장 움직임을 파악할 수 있어요."
                    )})
                else:
                    return JsonResponse({"response": "번호를 정확히 입력해주세요! (1~4)"})

            if any(kw in msg for kw in ["사용법", "사이트 사용법", "사이트 사용법을 알려줘", "이 사이트 어떻게 써요?"]):
                session["mode"] = "usage_selection"
                return JsonResponse({"response": (
                    "🤖 어떤 사용법이 궁금하신가요? 아래 중 번호를 입력해 주세요!\n\n"
                    "1. Fivo 소개\n"
                    "2. 주식 종목별 주가 예측 이용법\n"
                    "3. 트리맵 보는 법\n"
                    "4. 캔들 차트 보는 법"
                )})

            if any(kw in msg for kw in ["안녕", "안녕하세요", "하이"]):
                return JsonResponse({"response": "🤖 안녕하세요! 무엇을 도와드릴까요?"})

            if any(kw in msg for kw in ["오늘의", "기피", "유망"]):
                return JsonResponse({"response": "🤖 오늘의 유망&기피 종목 서비스는 아직 준비중입니다.\n불편을 끼쳐드려 죄송합니다."})

            intent_keywords = ["예측", "괜찮아", "유망", "어때", "왜 올라", "왜 떨어져", "전망", "분석"]

            stock_name, suggestions = extract_stock_name(msg)
            print("🧪 stock_name:", stock_name)
            print("🧪 suggestions:", suggestions)

            # 1️⃣ 종목명을 정확히 찾았을 때
            if stock_name:
                # 👉 의도 키워드가 없으면 유도 질문
                if not any(kw in msg for kw in intent_keywords):
                    return JsonResponse({
                        "response": f"'{stock_name}'에 대해 어떤 정보가 궁금하신가요? 😊\n예: '{stock_name} 주가 예측해줘', '{stock_name} 왜 떨어졌어?'"
                    })

                # 👉 의도 키워드까지 있으면 분석 진행
                news_articles = get_latest_news(stock_name)

                news_probs_list = []
                pred_classes = []

                for article in news_articles:
                    preds, probs = predict(article)
                    pred_class = preds[0]
                    prob_dist = probs[0][:4]

                    pred_classes.append(pred_class)
                    news_probs_list.append(prob_dist)

                if not news_probs_list:
                    return JsonResponse({"response": "🤖 관련 뉴스를 찾지 못했어요 😥"})

                avg_probs = [sum(p[i] for p in news_probs_list) / len(news_probs_list) for i in range(4)]
                predicted_class = avg_probs.index(max(avg_probs))

                print(avg_probs)
                print(predicted_class)
                exit() # ← 개발 중이 아니면 제거해도 돼

                risk_data = get_financial_risks(stock_name)
                score = calculate_final_score(avg_probs, risk_data)
                interpretation = interpret_score(score)
                summary = generate_news_summary(news_articles, predicted_class)

                return JsonResponse({
                    "response": f"🤖 {stock_name} 종목에 대한 분석 결과예요 📈",
                    "interpretation": interpretation,
                    "summary": summary
                })

            # 2️⃣ 종목을 정확히 찾지 못했지만 유사한 후보가 있을 경우
            elif suggestions:
                return JsonResponse({
                    "response": f"🤖 입력하신 종목명을 정확히 찾을 수 없어요.\n혹시 아래 중 하나인가요?",
                    "suggestions": suggestions
                })

            # 3️⃣ 아무것도 못 찾았을 때
            else:
                return JsonResponse({
                    "response": "🤖 관련된 종목명을 찾지 못했어요 😢"
                })


        return JsonResponse({"error": "POST 요청만 허용됩니다."}, status=405)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"서버 오류: {str(e)}"}, status=500)
