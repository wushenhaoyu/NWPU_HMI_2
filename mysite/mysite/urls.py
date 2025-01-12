"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp.views import video_feed,turn_camera,turn_face,turn_align,turn_point,storage_face
urlpatterns = [
    path("admin/", admin.site.urls),
    path("video", video_feed, name="video"),#output 
    path("turn_camera", turn_camera, name="turn"),
    path("turn_face", turn_face, name="turn_face"),
    path("turn_align", turn_align, name="turn_align"),
    path("turn_point", turn_point, name="turn_point"),
    path("storage_face", storage_face, name="storage_face"),

]
