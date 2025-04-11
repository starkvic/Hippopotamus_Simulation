from django.shortcuts import render

from simulation_app.management.commands.run_algorithm import HippopotamusAlgorithm
from .forms import SimulationSettingsForm

# Import your simulation function, for example:
# from .simulation_algorithms import HippopotamusAlgorithm

def simulation_settings(request):
    """
    View to display the simulation settings form and run the simulation.
    """
    if request.method == "POST":
        form = SimulationSettingsForm(request.POST)
        if form.is_valid():
            # Extract parameters from the form
            pop_size = form.cleaned_data['pop_size']
            max_iter = form.cleaned_data['max_iter']
            initial_irradiance = form.cleaned_data['initial_irradiance']
            T_value = form.cleaned_data['T_value']
            
            # (Example) Run the simulation with the new parameters.
            # Here, we're calling one of your algorithms.
            # You can choose to run all algorithms or a specific one.
            hippo_results = HippopotamusAlgorithm(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
            # Optionally, process and prepare the results (e.g., best voltage, power, convergence)
            
            # Pass simulation results to a results template.
            context = {
                'results': hippo_results,
                # You might add other context variables as needed.
            }
            return render(request, 'simulation_app/results.html', context)
    else:
        form = SimulationSettingsForm()
    return render(request, 'simulation_app/settings.html', {'form': form})


# Create your views here.
def home(request):
    return render(request, 'simulation_app/home.html')

def simulation(request):
    return render(request, 'simulation_app/simulation.html')

def about(request):
    return render(request, 'simulation_app/about.html')

def results(request):
    return render(request, 'simulation_app/results.html')