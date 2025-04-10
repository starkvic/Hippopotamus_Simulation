from django.urls import path
from .views import home, simulation, about  # assuming 'home' and 'about' are also defined

urlpatterns = [
    path('', home, name='home'),
    path('simulation/', simulation, name='simulation'),
    path('about/', about, name='about'),
]