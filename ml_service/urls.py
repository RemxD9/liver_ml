from django.contrib import admin
from django.urls import path
from Models_app import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main_page, name='main'),
    path('watch/', views.watching_photos, name='watching_photos'),
    path('predict/', views.predict, name='predict'),
    path('results/', views.results, name='results'),
    path('clear-input/', views.clear_input, name='clear_input'),
    path('clear-output/', views.clear_output, name='clear_output'),
]
