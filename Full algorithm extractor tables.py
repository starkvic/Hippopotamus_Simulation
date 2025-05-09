import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import gamma, sin, pi

# Set random seed for reproducibility
np.random.seed(49)

# ------------------------------------------------------------------------------
# Parameter Definitions
# ------------------------------------------------------------------------------
pop_size = 30
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

# ------------------------------------------------------------------------------
# 1️⃣ PV System Model and Objective Function (Common)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 1.a Levy Flight Function for Cuckoo Search
# ------------------------------------------------------------------------------
def levy_flight(beta):
    sigma_u = (gamma(1+beta) * sin(pi*beta/2) / (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.randn() * sigma_u
    v = np.random.randn()
    step = u / (abs(v)**(1/beta))
    return step

# ------------------------------------------------------------------------------
# Helper Function: Export CSV Results
# ------------------------------------------------------------------------------
def export_csv_results(data_records, filename):
    df = pd.DataFrame(data_records)
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

# ------------------------------------------------------------------------------
# 2️⃣ Hippopotamus Algorithm for MPPT
# ------------------------------------------------------------------------------
def HippopotamusAlgorithm(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(ind)], G, T) for ind in population.flatten()], dtype=float)
    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    convergence, V_history, G_history, T_history = [], [], [], []
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
        G_history.append(G)
        T_history.append(T)
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
    return best_solution, best_fitness, convergence, V_history, G_history, T_history

# ------------------------------------------------------------------------------
# 3️⃣ TLBO for MPPT
# ------------------------------------------------------------------------------
def TLBO(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(ind)], G, T) for ind in population.flatten()], dtype=float)
    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    convergence, V_history, G_history, T_history = [], [], [], []
    for iteration in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        teacher = population[np.argmax(fitness)]
        mean = np.mean(population, axis=0)
        TF = np.random.randint(1, 3)
        for i in range(pop_size):
            new_solution = population[i] + np.random.rand() * (teacher - TF * mean)
            new_fitness = objective_function([float(new_solution[0])], G, T)
            if new_fitness > fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
        best_solution = population[np.argmax(fitness)]
        best_fitness = np.max(fitness)
        convergence.append(best_fitness)
        V_history.append(best_solution[0])
        G_history.append(G)
        T_history.append(T)
        if export_csv:
            data_records.append({
                "Iteration": iteration + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": best_solution[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "tlbo_detailed_results.csv")
    return best_solution, best_fitness, convergence, V_history, G_history, T_history

# ------------------------------------------------------------------------------
# 4️⃣ Genetic Algorithm (GA) for MPPT
# ------------------------------------------------------------------------------
def GeneticAlgorithm(pop_size, max_iter, G, T, mutation_rate=0.1, crossover_rate=0.8, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(ind)], G, T) for ind in population.flatten()], dtype=float)
    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    convergence, V_history = [], []
    G_history = [G] * max_iter
    T_history = [T] * max_iter
    for iteration in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        new_population = []
        while len(new_population) < pop_size:
            if np.random.rand() < crossover_rate:
                parents = population[np.random.choice(pop_size, 2, replace=False)]
                crossover_point = np.random.rand()
                child1 = crossover_point * parents[0] + (1 - crossover_point) * parents[1]
                child2 = (1 - crossover_point) * parents[0] + crossover_point * parents[1]
                new_population.extend([child1, child2])
            else:
                new_population.append(population[np.random.randint(0, pop_size)])
        new_population = np.array(new_population)[:pop_size]
        mutation_mask = np.random.rand(*new_population.shape) < mutation_rate
        mutation_values = np.random.uniform(-2, 2, new_population.shape) * mutation_mask
        new_population += mutation_values
        new_fitness = np.array([objective_function([float(ind)], G, T) for ind in new_population.flatten()], dtype=float)
        population = new_population
        fitness = new_fitness
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
        export_csv_results(data_records, "genetic_detailed_results.csv")
    return best_solution, best_fitness, convergence, V_history, G_history, T_history

# ------------------------------------------------------------------------------
# 5️⃣ Particle Swarm Optimization (PSO) for MPPT
# ------------------------------------------------------------------------------
def PSO_MPPT(num_particles, max_iter, G_init, T, export_csv=False):
    data_records = [] if export_csv else None
    w = 0.7; c1 = 1.5; c2 = 1.5
    positions = np.random.uniform(0, 100, (num_particles, 1))
    velocities = np.random.uniform(-10, 10, (num_particles, 1))
    G_current = G_init
    pbest_positions = positions.copy()
    pbest_fitness = np.array([objective_function([float(x)], G_current, T) for x in positions.flatten()], dtype=float)
    gbest_index = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]
    convergence, voltage_history = [], []
    irradiance_history, temperature_history = [], []
    
    for iteration in range(max_iter):
        G_current = G_current * (0.9 + 0.2 * np.random.rand())
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], 0, 100)
            fitness_val = objective_function([float(positions[i][0])], G_current, T)
            if fitness_val > pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = fitness_val
        current_gbest_index = np.argmax(pbest_fitness)
        if pbest_fitness[current_gbest_index] > gbest_fitness:
            gbest_position = pbest_positions[current_gbest_index].copy()
            gbest_fitness = pbest_fitness[current_gbest_index]
        convergence.append(gbest_fitness)
        voltage_history.append(gbest_position[0])
        irradiance_history.append(G_current)
        temperature_history.append(T)
        if export_csv:
            data_records.append({
                "Iteration": iteration + 1,
                "Best Power Output": gbest_fitness,
                "Best Voltage": gbest_position[0],
                "Fitness Array": pbest_fitness.tolist(),
                "Population Array": positions.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "pso_detailed_results.csv")
    return gbest_position, gbest_fitness, convergence, voltage_history, irradiance_history, temperature_history

# ------------------------------------------------------------------------------
# 6️⃣ Artificial Bee Colony (ABC) for MPPT
# ------------------------------------------------------------------------------
def ABC_MPPT(num_food_sources, max_iter, G, T, limit=20, export_csv=False):
    data_records = [] if export_csv else None
    food_sources = np.random.uniform(0, 100, (num_food_sources, 1))
    fitness = np.array([objective_function([float(x)], G, T) for x in food_sources.flatten()], dtype=float)
    trial = np.zeros(num_food_sources)
    best_index = np.argmax(fitness)
    gbest_position = food_sources[best_index].copy()
    gbest_fitness = fitness[best_index]
    
    convergence, voltage_history = [], []
    irradiance_history, T_history = [], []
    current_G = G
    for iter in range(max_iter):
        for i in range(num_food_sources):
            k = np.random.choice([j for j in range(num_food_sources) if j != i])
            phi = np.random.uniform(-1, 1)
            new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            new_solution = np.clip(new_solution, 0, 100)
            new_fitness = objective_function([float(new_solution[0])], current_G, T)
            if new_fitness > fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1
        if np.sum(fitness[fitness > -np.inf]) == 0:
            probabilities = np.ones(num_food_sources) / num_food_sources
        else:
            probabilities = fitness / np.sum(fitness)
        for i in range(num_food_sources):
            if np.random.rand() < probabilities[i]:
                k = np.random.choice([j for j in range(num_food_sources) if j != i])
                phi = np.random.uniform(-1, 1)
                new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                new_solution = np.clip(new_solution, 0, 100)
                new_fitness = objective_function([float(new_solution[0])], current_G, T)
                if new_fitness > fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1
        for i in range(num_food_sources):
            if trial[i] > limit:
                food_sources[i] = np.random.uniform(0, 100, (1,))
                fitness[i] = objective_function([float(food_sources[i][0])], current_G, T)
                trial[i] = 0
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > gbest_fitness:
            gbest_position = food_sources[current_best_index].copy()
            gbest_fitness = fitness[current_best_index]
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        convergence.append(gbest_fitness)
        voltage_history.append(gbest_position[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": iter + 1,
                "Best Power Output": gbest_fitness,
                "Best Voltage": gbest_position[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": food_sources.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "abc_detailed_results.csv")
    return gbest_position, gbest_fitness, convergence, voltage_history, irradiance_history, T

# ------------------------------------------------------------------------------
# 7️⃣ SA for MPPT
# ------------------------------------------------------------------------------
def SA_MPPT(max_iter, G_init, T, initial_temp_sa=50, cooling_rate=0.95, export_csv=False):
    data_records = [] if export_csv else None
    current_solution = np.array([np.random.uniform(0, 100)])
    current_fitness = objective_function([float(current_solution[0])], G_init, T)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    current_temp = initial_temp_sa
    convergence, voltage_history = [], []
    irradiance_history, T_history = [], []
    current_G = G_init
    for iteration in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        delta = np.random.uniform(-5, 5)
        candidate = current_solution + delta
        candidate = np.clip(candidate, 0, 100)
        candidate_fitness = objective_function([float(candidate[0])], current_G, T)
        delta_fitness = candidate_fitness - current_fitness
        if delta_fitness >= 0:
            current_solution = candidate
            current_fitness = candidate_fitness
        else:
            if np.random.rand() < np.exp(delta_fitness / current_temp):
                current_solution = candidate
                current_fitness = candidate_fitness
        if current_fitness > best_fitness:
            best_solution = current_solution.copy()
            best_fitness = current_fitness
        convergence.append(best_fitness)
        voltage_history.append(best_solution[0])
        irradiance_history.append(current_G)
        T_history.append(T)
        if export_csv:
            data_records.append({
                "Iteration": iteration + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": best_solution[0],
                "Fitness Array": [current_fitness],
                "Population Array": [float(current_solution[0])]
            })
        current_temp *= cooling_rate
    if export_csv:
        export_csv_results(data_records, "sa_detailed_results.csv")
    return best_solution, best_fitness, convergence, voltage_history, irradiance_history, T_history

# ------------------------------------------------------------------------------
# 8️⃣ GWO for MPPT
# ------------------------------------------------------------------------------
def GWO_MPPT(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    positions = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(ind)], G, T) for ind in positions.flatten()], dtype=float)
    indices = np.argsort(fitness)[::-1]
    alpha_position = positions[indices[0]].copy()
    alpha_fitness = fitness[indices[0]]
    beta_position = positions[indices[1]].copy() if pop_size > 1 else alpha_position
    delta_position = positions[indices[2]].copy() if pop_size > 2 else beta_position
    convergence, voltage_history = [], []
    irradiance_history, T_history = [], []
    current_G = G
    for iteration in range(max_iter):
        a = 2 - iteration * (2 / max_iter)
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_position - positions[i])
            X1 = alpha_position - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_position - positions[i])
            X2 = beta_position - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_position - positions[i])
            X3 = delta_position - A3 * D_delta

            positions[i] = (X1 + X2 + X3) / 3.0
            positions[i] = np.clip(positions[i], 0, 100)
        fitness = np.array([objective_function([float(ind)], current_G, T) for ind in positions.flatten()], dtype=float)
        indices = np.argsort(fitness)[::-1]
        alpha_position = positions[indices[0]].copy()
        alpha_fitness = fitness[indices[0]]
        if pop_size > 1:
            beta_position = positions[indices[1]].copy()
        if pop_size > 2:
            delta_position = positions[indices[2]].copy()
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        convergence.append(alpha_fitness)
        voltage_history.append(alpha_position[0])
        irradiance_history.append(current_G)
        T_history.append(T)
        if export_csv:
            data_records.append({
                "Iteration": iteration + 1,
                "Best Power Output": alpha_fitness,
                "Best Voltage": alpha_position[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": positions.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "gwo_detailed_results.csv")
    return alpha_position, alpha_fitness, convergence, voltage_history, irradiance_history, T_history

# ------------------------------------------------------------------------------
# 9️⃣ Harmony Search (HS) for MPPT
# ------------------------------------------------------------------------------
def HarmonySearchMPPT(hm_size, max_iter, G_init, T, HMCR=0.9, PAR=0.3, bw=5, export_csv=False):
    data_records = [] if export_csv else None
    harmony_memory = np.random.uniform(0, 100, (hm_size, 1))
    fitness_memory = np.array([objective_function([float(harmony_memory[i])], G_init, T) for i in range(hm_size)], dtype=float)
    idx_sorted = np.argsort(fitness_memory)[::-1]
    harmony_memory = harmony_memory[idx_sorted, :]
    fitness_memory = fitness_memory[idx_sorted]
    
    best_harmony = harmony_memory[0].copy()
    best_fitness = fitness_memory[0]
    
    convergence, voltage_history, irradiance_history = [], [], []
    current_G = G_init
    for it in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        new_harmony = np.zeros((1,))
        if np.random.rand() < HMCR:
            new_harmony[0] = np.random.choice(harmony_memory.flatten())
            if np.random.rand() < PAR:
                new_harmony[0] += np.random.uniform(-bw, bw)
        else:
            new_harmony[0] = np.random.uniform(0, 100)
        new_harmony[0] = np.clip(new_harmony[0], 0, 100)
        new_fitness = objective_function([float(new_harmony[0])], current_G, T)
        worst_index = np.argmin(fitness_memory)
        if new_fitness > fitness_memory[worst_index]:
            harmony_memory[worst_index] = new_harmony
            fitness_memory[worst_index] = new_fitness
            idx_sorted = np.argsort(fitness_memory)[::-1]
            harmony_memory = harmony_memory[idx_sorted, :]
            fitness_memory = fitness_memory[idx_sorted]
        if fitness_memory[0] > best_fitness:
            best_harmony = harmony_memory[0].copy()
            best_fitness = fitness_memory[0]
        convergence.append(best_fitness)
        voltage_history.append(best_harmony[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": it + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": best_harmony[0],
                "Fitness Array": fitness_memory.tolist(),
                "Population Array": harmony_memory.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "harmony_detailed_results.csv")
    return best_harmony, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  🔟 Clonal Selection (CSA) for MPPT
# ------------------------------------------------------------------------------
def ClonalSelectionMPPT(pop_size, max_iter, G_init, T, clone_factor=5, mutation_rate=0.2, replacement_rate=0.2, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(v)], G_init, T) for v in population.flatten()], dtype=float)
    
    best_index = np.argmax(fitness)
    best_solution = population[best_index].copy()
    best_fitness = fitness[best_index]
    
    convergence, voltage_history, irradiance_history = [], [], []
    current_G = G_init
    
    for iter in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        clones = []
        for rank, candidate in enumerate(population):
            n_clones = int(np.ceil(clone_factor * (pop_size - rank) / pop_size))
            for _ in range(n_clones):
                clones.append(candidate.copy())
        clones = np.array(clones)
        mutated_clones = clones + np.random.uniform(-mutation_rate * 100, mutation_rate * 100, clones.shape)
        mutated_clones = np.clip(mutated_clones, 0, 100)
        clone_fitness = np.array([objective_function([float(v)], current_G, T) for v in mutated_clones.flatten()], dtype=float)
        combined_population = np.vstack((population, mutated_clones))
        combined_fitness = np.concatenate((fitness, clone_fitness))
        best_indices = np.argsort(combined_fitness)[::-1][:pop_size]
        population = combined_population[best_indices]
        fitness = combined_fitness[best_indices]
        n_replace = int(np.ceil(replacement_rate * pop_size))
        if n_replace > 0:
            new_candidates = np.random.uniform(0, 100, (n_replace, 1))
            new_fitness = np.array([objective_function([float(v)], current_G, T) for v in new_candidates.flatten()], dtype=float)
            worst_indices = np.argsort(fitness)[:n_replace]
            population[worst_indices, :] = new_candidates.reshape(-1, 1)
            fitness[worst_indices] = new_fitness.reshape(-1)
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            best_solution = population[current_best_index].copy()
            best_fitness = fitness[current_best_index]
        convergence.append(best_fitness)
        voltage_history.append(best_solution[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": iter + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": best_solution[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "csa_detailed_results.csv")
    return best_solution, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣1️⃣ Locust Swarm Algorithm (LSA) for MPPT
# ------------------------------------------------------------------------------
def LocustSwarmMPPT(num_locusts, max_iter, G_init, T, export_csv=False):
    data_records = [] if export_csv else None
    positions = np.random.uniform(0, 100, (num_locusts, 1))
    fitness = np.array([objective_function([float(pos)], G_init, T) for pos in positions.flatten()], dtype=float)
    best_index = np.argmax(fitness)
    global_best = positions[best_index].copy()
    best_fitness = fitness[best_index]
    convergence, voltage_history, irradiance_history = [], [], []
    current_G = G_init
    alpha = 0.5; beta = 0.3; gamma = 0.2
    for it in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        swarm_mean = np.mean(positions)
        for i in range(num_locusts):
            r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
            new_position = positions[i] + alpha * r1 * (global_best - positions[i]) \
                           + beta * r2 * (swarm_mean - positions[i]) \
                           + gamma * r3 * 100 * (np.random.rand() - 0.5)
            positions[i] = np.clip(new_position, 0, 100)
        fitness = np.array([objective_function([float(pos)], current_G, T) for pos in positions.flatten()], dtype=float)
        best_index = np.argmax(fitness)
        if fitness[best_index] > best_fitness:
            global_best = positions[best_index].copy()
            best_fitness = fitness[best_index]
        convergence.append(best_fitness)
        voltage_history.append(global_best[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": it + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": global_best[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": positions.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "lsa_detailed_results.csv")
    return global_best, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣2️⃣ Emperor Penguin Optimizer (EPO) for MPPT
# ------------------------------------------------------------------------------
def EmperorPenguinOptimizer(pop_size, max_iter, G_init, T, export_csv=False):
    data_records = [] if export_csv else None
    positions = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(x)], G_init, T) for x in positions.flatten()], dtype=float)
    best_index = np.argmax(fitness)
    global_best = positions[best_index].copy()
    best_fitness = fitness[best_index]
    convergence, voltage_history, irradiance_history = [], [], []
    current_G = G_init
    for it in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        a = 2 - it * (2 / max_iter)
        swarm_mean = np.mean(positions)
        for i in range(pop_size):
            r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
            new_position = positions[i] + a * r1 * (global_best - positions[i]) \
                           + 0.5 * r2 * (swarm_mean - positions[i]) \
                           + 0.1 * r3 * 100 * (np.random.rand() - 0.5)
            positions[i] = np.clip(new_position, 0, 100)
        fitness = np.array([objective_function([float(x)], current_G, T) for x in positions.flatten()], dtype=float)
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            global_best = positions[current_best_index].copy()
            best_fitness = fitness[current_best_index]
        convergence.append(best_fitness)
        voltage_history.append(global_best[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": it + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": global_best[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": positions.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "epo_detailed_results.csv")
    return global_best, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣3️⃣ Cuckoo Search (CS) for MPPT
# ------------------------------------------------------------------------------
def CuckooSearchMPPT(n_nests, max_iter, G_init, T, pa=0.25, alpha_cs=0.01, beta_cs=1.5, export_csv=False):
    data_records = [] if export_csv else None
    nests = np.random.uniform(0, 100, (n_nests, 1))
    fitness = np.array([objective_function([float(nest)], G_init, T) for nest in nests.flatten()], dtype=float)
    best_index = np.argmax(fitness)
    best_nest = nests[best_index].copy()
    best_fitness = fitness[best_index]
    convergence, voltage_history, irradiance_history = [], [], []
    current_G = G_init
    for t in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        new_nests = np.empty_like(nests)
        for i in range(n_nests):
            step = levy_flight(beta_cs)
            new_nest = nests[i] + alpha_cs * step
            new_nest = np.clip(new_nest, 0, 100)
            new_nests[i] = new_nest
        new_fitness = np.array([objective_function([float(nest)], current_G, T) for nest in new_nests.flatten()], dtype=float)
        for i in range(n_nests):
            j = np.random.randint(0, n_nests)
            if new_fitness[i] > fitness[j]:
                nests[j] = new_nests[i]
                fitness[j] = new_fitness[i]
        num_abandoned = int(pa * n_nests)
        worst_indices = np.argsort(fitness)[:num_abandoned]
        nests[worst_indices] = np.random.uniform(0, 100, (num_abandoned, 1))
        fitness[worst_indices] = np.array([objective_function([float(nest)], current_G, T) for nest in nests[worst_indices].flatten()], dtype=float)
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            best_nest = nests[current_best_index].copy()
            best_fitness = fitness[current_best_index]
        convergence.append(best_fitness)
        voltage_history.append(best_nest[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": t + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": best_nest[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": nests.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "cuckoo_detailed_results.csv")
    return best_nest, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣4️⃣ Artificial Butterfly Optimization (ABO) for MPPT
# ------------------------------------------------------------------------------
def ABO_MPPT(pop_size, max_iter, G_init, T, p=0.8, c=0.01, a_exp=0.1, export_csv=False):
    data_records = [] if export_csv else None
    positions = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function([float(x)], G_init, T) for x in positions.flatten()], dtype=float)
    best_index = np.argmax(fitness)
    global_best = positions[best_index].copy()
    best_fitness = fitness[best_index]
    convergence, voltage_history, irradiance_history = [], [], []
    current_G = G_init
    for t in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        for i in range(pop_size):
            f_i = c * (fitness[i]**a_exp) if fitness[i] != -np.inf else 0
            r = np.random.rand()
            if r < p:
                new_position = positions[i] + r * (global_best - positions[i]) * f_i
            else:
                idxs = np.random.choice(range(pop_size), 2, replace=False)
                new_position = positions[i] + r * (positions[idxs[0]] - positions[idxs[1]]) * f_i
            positions[i] = np.clip(new_position, 0, 100)
        fitness = np.array([objective_function([float(x)], current_G, T) for x in positions.flatten()], dtype=float)
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            global_best = positions[current_best_index].copy()
            best_fitness = fitness[current_best_index]
        convergence.append(best_fitness)
        voltage_history.append(global_best[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": t + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": global_best[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": positions.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "abo_detailed_results.csv")
    return global_best, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣5️⃣ Human Evolutionary Model (HEM) for MPPT
# ------------------------------------------------------------------------------
def HEM_MPPT(pop_size, max_iter, G_init, T, selection_rate=0.3, mutation_rate=0.1, crossover_rate=0.7, diversity_rate=0.1, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0, 100, (pop_size, 1))
    current_G = G_init
    fitness = np.array([objective_function([float(x)], current_G, T) for x in population.flatten()], dtype=float)
    best_index = np.argmax(fitness)
    global_best = population[best_index].copy()
    best_fitness = fitness[best_index]
    convergence, voltage_history, irradiance_history = [], [], []
    for it in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        elite_count = int(np.ceil(selection_rate * pop_size))
        sorted_indices = np.argsort(fitness)[::-1]
        elites = population[sorted_indices[:elite_count]]
        children = []
        while len(children) < pop_size - elite_count:
            if np.random.rand() < crossover_rate:
                parents = elites[np.random.choice(elite_count, 2, replace=False)]
                weight = np.random.rand()
                child = weight * parents[0] + (1 - weight) * parents[1]
            else:
                child = elites[np.random.randint(0, elite_count)]
            child += np.random.uniform(-mutation_rate * 100, mutation_rate * 100, child.shape)
            child = np.clip(child, 0, 100)
            children.append(child)
        children = np.array(children)
        r = np.random.rand(*children.shape)
        children = children + r * (global_best - children)
        children = np.clip(children, 0, 100)
        diversity_count = int(np.ceil(diversity_rate * pop_size))
        random_individuals = np.random.uniform(0, 100, (diversity_count, 1))
        new_pop = np.vstack((elites, children))
        if new_pop.shape[0] > pop_size:
            new_pop = new_pop[:pop_size]
        else:
            new_pop[-diversity_count:] = random_individuals
        population = new_pop
        fitness = np.array([objective_function([float(x)], current_G, T) for x in population.flatten()], dtype=float)
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            global_best = population[current_best_index].copy()
            best_fitness = fitness[current_best_index]
        convergence.append(best_fitness)
        voltage_history.append(global_best[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": it + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": global_best[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "hem_detailed_results.csv")
    return global_best, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣6️⃣ White Shark Optimizer (WSO) for MPPT
# ------------------------------------------------------------------------------
def WhiteSharkOptimizerMPPT(pop_size, max_iter, G_init, T, export_csv=False):
    data_records = [] if export_csv else None
    positions = np.random.uniform(0, 100, (pop_size, 1))
    current_G = G_init
    fitness = np.array([objective_function([float(x)], current_G, T) for x in positions.flatten()], dtype=float)
    best_index = np.argmax(fitness)
    global_best = positions[best_index].copy()
    best_fitness = fitness[best_index]
    convergence, voltage_history, irradiance_history = [], [], []
    for it in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        hunting_factor = np.exp(-it / max_iter)
        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            new_position = positions[i] + r1 * hunting_factor * (global_best - positions[i]) + \
                           r2 * (np.random.rand() - 0.5) * 10
            positions[i] = np.clip(new_position, 0, 100)
        fitness = np.array([objective_function([float(x)], current_G, T) for x in positions.flatten()], dtype=float)
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            global_best = positions[current_best_index].copy()
            best_fitness = fitness[current_best_index]
        convergence.append(best_fitness)
        voltage_history.append(global_best[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": it + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": global_best[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": positions.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "wso_detailed_results.csv")
    return global_best, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣7️⃣ Differential Evolution (DE) for MPPT
# ------------------------------------------------------------------------------
def DifferentialEvolutionMPPT(pop_size, max_iter, G_init, T, F=0.8, CR=0.9, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0, 100, (pop_size, 1))
    current_G = G_init
    fitness = np.array([objective_function([float(x)], current_G, T) for x in population.flatten()], dtype=float)
    best_index = np.argmax(fitness)
    global_best = population[best_index].copy()
    best_fitness = fitness[best_index]
    convergence, voltage_history, irradiance_history = [], [], []
    for it in range(max_iter):
        current_G = current_G * (0.9 + 0.2 * np.random.rand())
        new_population = np.empty_like(population)
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            r = np.random.choice(indices, 3, replace=False)
            r1, r2, r3 = population[r[0]], population[r[1]], population[r[2]]
            mutant = r1 + F * (r2 - r3)
            if np.random.rand() < CR:
                trial = mutant
            else:
                trial = population[i]
            new_population[i] = np.clip(trial, 0, 100)
        new_fitness = np.array([objective_function([float(x)], current_G, T) for x in new_population.flatten()], dtype=float)
        for i in range(pop_size):
            if new_fitness[i] > fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]
        best_index = np.argmax(fitness)
        if fitness[best_index] > best_fitness:
            global_best = population[best_index].copy()
            best_fitness = fitness[best_index]
        convergence.append(best_fitness)
        voltage_history.append(global_best[0])
        irradiance_history.append(current_G)
        if export_csv:
            data_records.append({
                "Iteration": it + 1,
                "Best Power Output": best_fitness,
                "Best Voltage": global_best[0],
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "de_detailed_results.csv")
    return global_best, best_fitness, convergence, voltage_history, irradiance_history

# ------------------------------------------------------------------------------
#  1️⃣8️⃣ Run All 16 MPPT Algorithms with CSV Export Enabled
# ------------------------------------------------------------------------------
hippo_results = HippopotamusAlgorithm(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
tlbo_results  = TLBO(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
ga_results    = GeneticAlgorithm(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
pso_results   = PSO_MPPT(num_particles, max_iter, initial_irradiance, T_value, export_csv=True)
abc_results   = ABC_MPPT(num_food_sources, max_iter, initial_irradiance, T_value, export_csv=True)
sa_results    = SA_MPPT(max_iter, initial_irradiance, T_value, initial_temp_sa, cooling_rate, export_csv=True)
gwo_results   = GWO_MPPT(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
hs_results    = HarmonySearchMPPT(hm_size, max_iter, initial_irradiance, T_value, HMCR, PAR, bw, export_csv=True)
csa_results   = ClonalSelectionMPPT(pop_size, max_iter, initial_irradiance, T_value, clone_factor, mutation_rate, replacement_rate, export_csv=True)
lsa_results   = LocustSwarmMPPT(num_locusts, max_iter, initial_irradiance, T_value, export_csv=True)
epo_results   = EmperorPenguinOptimizer(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
cs_results    = CuckooSearchMPPT(n_nests, max_iter, initial_irradiance, T_value, pa, alpha_cs, beta_cs, export_csv=True)
abo_results   = ABO_MPPT(pop_size, max_iter, initial_irradiance, T_value, p, c, a_exp, export_csv=True)
hem_results   = HEM_MPPT(pop_size, max_iter, initial_irradiance, T_value, 0.3, 0.1, 0.7, 0.1, export_csv=True)
wso_results   = WhiteSharkOptimizerMPPT(pop_size, max_iter, initial_irradiance, T_value, export_csv=True)
de_results    = DifferentialEvolutionMPPT(pop_size, max_iter, initial_irradiance, T_value, F, CR, export_csv=True)

# Unpack convergence arrays (for plotting and table)
convergence_hippo, V_hippo = hippo_results[2], hippo_results[3]
convergence_tlbo, V_tlbo   = tlbo_results[2], tlbo_results[3]
convergence_ga, V_ga       = ga_results[2], ga_results[3]
convergence_pso, V_pso     = pso_results[2], pso_results[3]
convergence_abc, V_abc     = abc_results[2], abc_results[3]
convergence_sa, V_sa       = sa_results[2], sa_results[3]
convergence_gwo, V_gwo     = gwo_results[2], gwo_results[3]
convergence_hs, V_hs       = hs_results[2], hs_results[3]
convergence_csa, V_csa     = csa_results[2], csa_results[3]
convergence_lsa, V_lsa     = lsa_results[2], lsa_results[3]
convergence_epo, V_epo     = epo_results[2], epo_results[3]
convergence_cs, V_cs       = cs_results[2], cs_results[3]
convergence_abo, V_abo     = abo_results[2], abo_results[3]
convergence_hem, V_hem     = hem_results[2], hem_results[3]
convergence_wso, V_wso     = wso_results[2], wso_results[3]
convergence_de, V_de       = de_results[2], de_results[3]

# ------------------------------------------------------------------------------
#  1️⃣9️⃣ Define Unique Markers and Colors for Each Algorithm
# ------------------------------------------------------------------------------
styles = {
    "Hippopotamus": {"marker": "x", "color": "blue"},
    "TLBO": {"marker": "s", "color": "red"},
    "Genetic Alg": {"marker": "^", "color": "green"},
    "PSO": {"marker": "o", "color": "purple"},
    "ABC": {"marker": "d", "color": "brown"},
    "SA": {"marker": "v", "color": "gray"},
    "GWO": {"marker": "p", "color": "magenta"},
    "HS": {"marker": "*", "color": "cyan"},
    "CSA": {"marker": ">", "color": "black"},
    "LSA": {"marker": "<", "color": "orange"},
    "EPO": {"marker": "h", "color": "lime"},
    "Cuckoo Search": {"marker": "D", "color": "gold"},
    "ABO": {"marker": "+", "color": "darkorange"},
    "HEM": {"marker": "H", "color": "darkblue"},
    "WSO": {"marker": "8", "color": "darkgreen"},
    "DE": {"marker": "X", "color": "saddlebrown"}
}

# ------------------------------------------------------------------------------
#  2️⃣0️⃣ Plot Performance Comparison (Five Key Plots)
# ------------------------------------------------------------------------------

# Plot 1: Convergence Curve (Power Output over Iterations)
plt.figure(figsize=(10, 6))
plt.plot(convergence_hippo, label="Hippopotamus", linestyle='--', marker=styles["Hippopotamus"]["marker"], color=styles["Hippopotamus"]["color"])
plt.plot(convergence_tlbo, label="TLBO", linestyle='-', marker=styles["TLBO"]["marker"], color=styles["TLBO"]["color"])
plt.plot(convergence_ga, label="Genetic Alg", linestyle='-.', marker=styles["Genetic Alg"]["marker"], color=styles["Genetic Alg"]["color"])
plt.plot(convergence_pso, label="PSO", linestyle=':', marker=styles["PSO"]["marker"], color=styles["PSO"]["color"])
plt.plot(convergence_abc, label="ABC", linestyle='-', marker=styles["ABC"]["marker"], color=styles["ABC"]["color"])
plt.plot(convergence_sa, label="SA", linestyle='--', marker=styles["SA"]["marker"], color=styles["SA"]["color"])
plt.plot(convergence_gwo, label="GWO", linestyle='-.', marker=styles["GWO"]["marker"], color=styles["GWO"]["color"])
plt.plot(convergence_hs, label="HS", linestyle='-', marker=styles["HS"]["marker"], color=styles["HS"]["color"])
plt.plot(convergence_csa, label="CSA", linestyle=':', marker=styles["CSA"]["marker"], color=styles["CSA"]["color"])
plt.plot(convergence_lsa, label="LSA", linestyle='-', marker=styles["LSA"]["marker"], color=styles["LSA"]["color"])
plt.plot(convergence_epo, label="EPO", linestyle='--', marker=styles["EPO"]["marker"], color=styles["EPO"]["color"])
plt.plot(convergence_cs, label="Cuckoo Search", linestyle='-.', marker=styles["Cuckoo Search"]["marker"], color=styles["Cuckoo Search"]["color"])
plt.plot(convergence_abo, label="ABO", linestyle='-', marker=styles["ABO"]["marker"], color=styles["ABO"]["color"])
plt.plot(convergence_hem, label="HEM", linestyle='--', marker=styles["HEM"]["marker"], color=styles["HEM"]["color"])
plt.plot(convergence_wso, label="WSO", linestyle=':', marker=styles["WSO"]["marker"], color=styles["WSO"]["color"])
plt.plot(convergence_de, label="DE", linestyle='-.', marker=styles["DE"]["marker"], color=styles["DE"]["color"])
plt.title("Convergence Curve (Power Output over Iterations)")
plt.xlabel("Iteration")
plt.ylabel("Power Output (W)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Voltage Evolution over Iterations
plt.figure(figsize=(10, 6))
plt.plot(V_hippo, label="Hippopotamus Voltage", linestyle='--', marker=styles["Hippopotamus"]["marker"], color=styles["Hippopotamus"]["color"])
plt.plot(V_tlbo, label="TLBO Voltage", linestyle='-', marker=styles["TLBO"]["marker"], color=styles["TLBO"]["color"])
plt.plot(V_ga, label="Genetic Alg Voltage", linestyle='-.', marker=styles["Genetic Alg"]["marker"], color=styles["Genetic Alg"]["color"])
plt.plot(V_pso, label="PSO Voltage", linestyle=':', marker=styles["PSO"]["marker"], color=styles["PSO"]["color"])
plt.plot(V_abc, label="ABC Voltage", linestyle='-', marker=styles["ABC"]["marker"], color=styles["ABC"]["color"])
plt.plot(V_sa, label="SA Voltage", linestyle='--', marker=styles["SA"]["marker"], color=styles["SA"]["color"])
plt.plot(V_gwo, label="GWO Voltage", linestyle='-.', marker=styles["GWO"]["marker"], color=styles["GWO"]["color"])
plt.plot(V_hs, label="HS Voltage", linestyle='-', marker=styles["HS"]["marker"], color=styles["HS"]["color"])
plt.plot(V_csa, label="CSA Voltage", linestyle=':', marker=styles["CSA"]["marker"], color=styles["CSA"]["color"])
plt.plot(V_lsa, label="LSA Voltage", linestyle='-', marker=styles["LSA"]["marker"], color=styles["LSA"]["color"])
plt.plot(V_epo, label="EPO Voltage", linestyle='--', marker=styles["EPO"]["marker"], color=styles["EPO"]["color"])
plt.plot(V_cs, label="Cuckoo Search Voltage", linestyle='-.', marker=styles["Cuckoo Search"]["marker"], color=styles["Cuckoo Search"]["color"])
plt.plot(V_abo, label="ABO Voltage", linestyle='-', marker=styles["ABO"]["marker"], color=styles["ABO"]["color"])
plt.plot(V_hem, label="HEM Voltage", linestyle='--', marker=styles["HEM"]["marker"], color=styles["HEM"]["color"])
plt.plot(V_wso, label="WSO Voltage", linestyle=':', marker=styles["WSO"]["marker"], color=styles["WSO"]["color"])
plt.plot(V_de, label="DE Voltage", linestyle='-.', marker=styles["DE"]["marker"], color=styles["DE"]["color"])
plt.title("Voltage Evolution over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.show()



# ------------------------------------------------------------------------------
#  2️⃣0️⃣ Create a Summary Table of Metrics for Each Algorithm
# ------------------------------------------------------------------------------
def get_metrics(name, convergence, voltage):
    iteration_reached = int(np.argmax(convergence))
    best_voltage = voltage[-1]
    best_power = convergence[-1]
    return {"Algorithm": name,
            "Best Voltage (V)": best_voltage,
            "Best Power (W)": best_power,
            "Iteration Reached": iteration_reached}

results = []
results.append(get_metrics("Hippopotamus", convergence_hippo, V_hippo))
results.append(get_metrics("TLBO", convergence_tlbo, V_tlbo))
results.append(get_metrics("Genetic Alg", convergence_ga, V_ga))
results.append(get_metrics("PSO", convergence_pso, V_pso))
results.append(get_metrics("ABC", convergence_abc, V_abc))
results.append(get_metrics("SA", convergence_sa, V_sa))
results.append(get_metrics("GWO", convergence_gwo, V_gwo))
results.append(get_metrics("HS", convergence_hs, V_hs))
results.append(get_metrics("CSA", convergence_csa, V_csa))
results.append(get_metrics("LSA", convergence_lsa, V_lsa))
results.append(get_metrics("EPO", convergence_epo, V_epo))
results.append(get_metrics("Cuckoo Search", convergence_cs, V_cs))
results.append(get_metrics("ABO", convergence_abo, V_abo))
results.append(get_metrics("HEM", convergence_hem, V_hem))
results.append(get_metrics("WSO", convergence_wso, V_wso))
results.append(get_metrics("DE", convergence_de, V_de))

df_results = pd.DataFrame(results)
print("Summary Table of MPPT Algorithms Metrics:")
print(df_results)

# ------------------------------------------------------------------------------
#  2️⃣1️⃣ Create Additional Table: Power vs. Iteration Values for Each Algorithm
# ------------------------------------------------------------------------------
iterations = list(range(1, max_iter+1))
power_table = pd.DataFrame({
    "Iteration": iterations,
    "Hippopotamus": convergence_hippo,
    "TLBO": convergence_tlbo,
    "Genetic Alg": convergence_ga,
    "PSO": convergence_pso,
    "ABC": convergence_abc,
    "SA": convergence_sa,
    "GWO": convergence_gwo,
    "HS": convergence_hs,
    "CSA": convergence_csa,
    "LSA": convergence_lsa,
    "EPO": convergence_epo,
    "Cuckoo Search": convergence_cs,
    "ABO": convergence_abo,
    "HEM": convergence_hem,
    "WSO": convergence_wso,
    "DE": convergence_de
})
print("\nPower vs Iteration Table:")
print(power_table)
power_table.to_csv("power_vs_iteration.csv", index=False)
print("Saved power_vs_iteration.csv")
