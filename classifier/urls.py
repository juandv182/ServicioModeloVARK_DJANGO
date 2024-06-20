
from django.urls import path
from .views import ClassifierView

urlpatterns = [
    path('predict', ClassifierView.as_view()),
]