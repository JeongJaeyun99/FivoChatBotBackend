# urls.py
from django.urls import path
from .views import log_news_click

urlpatterns = [
    path("news_click/", log_news_click),
]