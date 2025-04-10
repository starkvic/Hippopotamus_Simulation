from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'simulation_app/home.html')

def simulation(request):
    return render(request, 'simulation_app/simulation.html')

def about(request):
    return render(request, 'simulation_app/about.html')
