from django.contrib import admin
from django.urls import path
from Models_app.views import main_page, watching_photos, predict, results, clear_input, clear_output


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', main_page, name='main'),
    path('watch/', watching_photos, name='watching_photos'),
    path('predict/', predict, name='predict'),
    path('results/', results, name='results'),
    path('clear-input/', clear_input, name='clear_input'),
    path('clear-output/', clear_output, name='clear_output'),
]
