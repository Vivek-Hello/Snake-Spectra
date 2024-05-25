from django.urls import path
from . import views

urlpatterns = [
    path('',views.find_snake_by_attributes,name='find_snake_by_attributes')
]
