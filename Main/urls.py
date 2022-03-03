from django.contrib import admin
from django.urls import path
from .views import PerdictView
urlpatterns = [
    path('predict/', PerdictView.as_view()),
]