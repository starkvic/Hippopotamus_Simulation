# simulation_app/management/commands/run_simulation.py

from django.core.management.base import BaseCommand
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import gamma, sin, pi

# -------------------------------------------------------------------------------
# Parameter Definitions
# -------------------------------------------------------------------------------
pop_size = 10
max_iter = 100
initial_irradiance = 800  # [W/m²]
T_value = 25              # [°C]

num_particles = 30        # For PSO
num_food_sources = 30     # For ABC
num_locusts = 30          # For LSA

# For SA:
initial_temp_sa = 50
cooling_rate = 0.95

# For Harmony Search (HS):
hm_size = 30
HMCR = 0.9
PAR = 0.3
bw = 5

# For Clonal Selection (CSA):
clone_factor = 5
mutation_rate = 0.2
replacement_rate = 0.2

# For Cuckoo Search (CS):
n_nests = 30
pa = 0.25
alpha_cs = 0.01  # scaling factor for Lévy flight
beta_cs = 1.5    # Lévy exponent

# For Artificial Butterfly Optimization (ABO):
p = 0.8        # probability for global search
c = 0.01       # sensory modality constant
a_exp = 0.1    # power exponent

# For Differential Evolution (DE):
F = 0.8        # Mutation scaling factor
CR = 0.9       # Crossover probability

# -------------------------------------------------------------------------------
# Define Simulation Functions
# -------------------------------------------------------------------------------

def pv_system_model(V, G, T):
    I_sc = 10              # [A]
    V_oc = 100             # [V]
    Temp_coeff_V = -0.005  # [-/°C]
    T_ref = 25             # [°C]
    Max_efficiency = 0.85  # [unitless]
    V_oc_adjusted = V_oc * (1 + Temp_coeff_V * (T - T_ref))
    I = I_sc * (1 - np.exp(-V / V_oc_adjusted)) * (G / 1000)
    P = V * I
    P_max = Max_efficiency * V_oc_adjusted * I_sc
    return -np.inf if P > P_max else P

def objective_function(params, G, T):
    V = params[0]
    return -np.inf if V < 0 or V > 100 else pv_system_model(V, G, T)

def levy_flight(beta):
    sigma_u = (gamma(1+beta) * sin(pi*beta/2) /
               (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.randn() * sigma_u
    v = np.random.randn()
    return u / (abs(v)**(1/beta))

def export_csv_results(data_records, filename):
    df = pd.DataFrame(data_records)
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

# -------------------------------------------------------------------------------
# (Example) Hippopotamus Algorithm for MPPT
# -------------------------------------------------------------------------------
def HippopotamusAlgorithm(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(ind)], G, T) for ind in population.flatten()], dtype=float)
    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    convergence, V_history = [], []
    for iteration in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        for i in range(pop_size):
            partner_idx = np.random.choice([j for j in range(pop_size) if j != i])
            partner = population[partner_idx]
            if np.random.rand() < 0.5:
                new_solution = population[i] + np.random.uniform(-1, 1) * (partner - population[i])
            else:
                new_solution = population[i] + 0.5 * (best_solution - population[i])
            new_solution = np.clip(new_solution, 0, 100)
            new_fitness = objective_function([float(new_solution[0])], G, T)
            if new_fitness > fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
        best_solution = population[np.argmax(fitness)]
        best_fitness = np.max(fitness)
        convergence.append(best_fitness)
        V_history.append(best_solution[0])
        if export_csv:
            data_records.append({
                "Iteration": iteration + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": best_solution[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "hippopotamus_detailed_results.csv")
    return best_solution, best_fitness, convergence, V_history

# (Similarly, you can include other algorithm functions here.)

# -------------------------------------------------------------------------------
# Django Management Command
# -------------------------------------------------------------------------------
class Command(BaseCommand):
    help = "Runs the MPPT simulation metaheuristic algorithms and generates CSV outputs."

    def handle(self, *args, **options):
        self.stdout.write("Starting simulations...")
        
        # Run one of the simulation algorithms (you can uncomment or add the others as needed).
        hippo_results = HippopotamusAlgorithm(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
        
        # Optionally, unpack results if you want to print a summary:
        best_solution, best_fitness, convergence, V_history = hippo_results
        self.stdout.write(self.style.SUCCESS("Hippopotamus Algorithm Simulation Complete."))
        self.stdout.write(f"Best Voltage: {best_solution[0]:.2f} V, Best Power: {best_fitness:.2f} W")
        
        # Optionally, you can run all algorithms and create summary plots or tables.
        # (You may want to remove or comment out plt.show() calls in production.)
        
        # For example, plot the convergence curve (this will open a window if run locally
        # but might not be suitable on a headless production server):
        plt.figure(figsize=(10, 6))
        plt.plot(convergence, label="Hippopotamus", linestyle='--', marker='x', color='blue')
        plt.title("Convergence Curve (Power Output over Iterations)")
        plt.xlabel("Iteration")
        plt.ylabel("Power Output (W)")
        plt.legend()
        # Instead of plt.show(), you can save the figure:
        plt.savefig("static/simulation_app/results/hippopotamus_convergence.png")
        self.stdout.write("Convergence plot saved as hippopotamus_convergence.png")
        
        self.stdout.write(self.style.SUCCESS("Simulations complete."))
