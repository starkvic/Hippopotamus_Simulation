from django.urls import path
from .api_views import simulation_status

urlpatterns = [
    path('simulation-status/', simulation_status, name='simulation_status'),
]
