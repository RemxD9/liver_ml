from django.contrib import admin
from django.urls import path
from Models_app import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main_page, name='main'),
    path('watch/', views.watching_photos, name='watching_photos'),
    path('predict/', views.predict, name='predict'),
    path('results/', views.results, name='results'),
    path('clear-input/', views.clear_input, name='clear_input'),
    path('clear-output/', views.clear_output, name='clear_output'),
    path('annotation/', views.annotation, name='annotation'),
    path('save_mask/', views.save_mask, name='save_mask'),
    path('download_file/', views.download_file, name='download_file'),
    path('add_mask/', views.add_mask, name='add_mask'),
    path('del_mask/', views.del_mask, name='del_mask'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
