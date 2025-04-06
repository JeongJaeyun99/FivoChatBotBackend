import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options  # ✅ 추가
import json
import time
import os

def latest_news(stock_name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "stock_list.json")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        stock_data = json.load(f)

    stock_code = None
    for item in stock_data:
        if item["회사명"] == stock_name:
            stock_code = item["종목코드"]
            break

    if not stock_code:
        print(f"❌ 종목명 '{stock_name}'을 찾을 수 없습니다.")
        return []

    url = f'https://finance.finup.co.kr/Stock/{stock_code}'

    # ✅ Headless Chrome 옵션 설정
    options = Options()
    options.add_argument("--headless")  # 창 안 뜨게!
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(2)  # JavaScript 로딩 대기

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    news_list = soup.select("div#divContents01 div#newsTab ul#ulStockDetailNews")

    news_items = []
    for li in news_list:
        for item in li.find_all("li"):
            title_tag = item.find("p", class_="mt5 cm_txt")
            summary_tag = item.find("p", class_="mt5 cm_smtxt cm_color_lg")
            if title_tag and summary_tag:
                title = title_tag.get_text(strip=True)
                summary = summary_tag.get_text(strip=True)
                news_items.append({
                    "title": title,
                    "summary": summary
                })

    driver.quit()
    return news_items

if __name__ == "__main__":
    # 종목이름을 불러와서 종목 코드를 불러온뒤 latest_news에 넣는다
    news = latest_news("DSR제강")
    for n in news:
        print("📰", n)
    print(len(news))