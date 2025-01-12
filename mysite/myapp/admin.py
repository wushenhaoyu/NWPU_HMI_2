from django.contrib import admin
from .models import Face  # 导入你的模型类

# 注册模型，使其在 Django Admin 中可见
admin.site.register(Face)
