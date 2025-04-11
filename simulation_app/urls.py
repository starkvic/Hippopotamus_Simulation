from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('simulation/', views.simulation, name='simulation'),
    path('about/', views.about, name='about'),
    path('results/', views.results, name='results'),  # Added results page
    path('settings/', views.simulation_settings, name='settings'),
]
