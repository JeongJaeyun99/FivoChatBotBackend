from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import NewsClick
import json

@csrf_exempt
def log_news_click(request):
    if request.method == "POST":
        data = json.loads(request.body)
        stock_name = data.get("stock_name")
        title = data.get("title")
        session_id = request.session.session_key
        ip = request.META.get("REMOTE_ADDR")

        NewsClick.objects.create(
            stock_name=stock_name,
            title=title,
            session_id=session_id,
            ip_address=ip
        )
        return JsonResponse({"status": "ok"})

    return JsonResponse({"error": "Invalid request"}, status=400)
