from django.db import models
from django.utils import timezone

class NewsClick(models.Model):
    stock_name = models.CharField(max_length=100)
    title = models.TextField()
    clicked_at = models.DateTimeField(default=timezone.now)
    session_id = models.CharField(max_length=100, blank=True, null=True)  # 선택
    ip_address = models.GenericIPAddressField(blank=True, null=True)

    def __str__(self):
        return f"[{self.clicked_at}] {self.stock_name} - {self.title}"
