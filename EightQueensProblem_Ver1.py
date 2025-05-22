import random
import time
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration & GA Parameters ---
N = 8  # Board size (8×8)
POPULATION_SIZE = 10  # Number of candidate solutions each generation
MAX_GENERATIONS = 500  # Maximum number of generations to evolve
MUTATION_RATE = 0.2   # Probability of swapping two genes
TOURNAMENT_SIZE = 5   # Number of individuals in each tournament for selection
# Delay between rendering each queen placement
VISUALIZATION_UPDATE_INTERVAL = 0.05  

# --- Helper Functions ---
def calculate_fitness(chromosome):
    """
    Fitness = number of non-attacking pairs of queens. 
    Maximum possible = N*(N-1)/2 (28 for N=8).
    We count diagonal conflicts and subtract from max.
    """
    max_pairs = N * (N - 1) // 2
    attacks = 0
    # Count diagonal attacks
    for i in range(N):
        for j in range(i + 1, N):
            if abs(i - j) == abs(chromosome[i] - chromosome[j]):
                attacks += 1
    return max_pairs - attacks


def create_chromosome():
    """
    Generate a random chromosome: a permutation of rows 0..N-1.
    Each index = column, value = row of the queen.
    Starting the GA with random positions is standard for POC.
    """
    perm = list(range(N))
    random.shuffle(perm)
    return perm

# --- Genetic Operators ---
def tournament_selection(population, fitness_scores):
    """
    Select one parent via tournament selection:
    pick TOURNAMENT_SIZE random contestants and choose the fittest.
    """
    # Pair each individual with its fitness for comparison
    contestants = random.sample(list(zip(population, fitness_scores)), TOURNAMENT_SIZE)
    # Return only the chromosome (not the fitness)
    return max(contestants, key=lambda pair: pair[1])[0]


def crossover(parent1, parent2):
    """
    Order Crossover (OX1):
    - Choose a subsequence from each parent to copy to child
    - Fill remaining positions in child in order from the other parent
    Produces two offspring.
    """
    size = N
    c1, c2 = [-1]*size, [-1]*size
    a, b = sorted(random.sample(range(size), 2))
    # Copy slice from parents
    c1[a:b+1] = parent1[a:b+1]
    c2[a:b+1] = parent2[a:b+1]
    # Fill the rest
    fill1 = [gene for gene in parent2 if gene not in c1]
    fill2 = [gene for gene in parent1 if gene not in c2]
    idx1 = b + 1
    for gene in fill1:
        c1[idx1 % size] = gene
        idx1 += 1
    idx2 = b + 1
    for gene in fill2:
        c2[idx2 % size] = gene
        idx2 += 1
    return c1, c2


def mutate(chromosome):
    """
    Swap mutation: with probability MUTATION_RATE, pick two positions and swap them.
    """
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(N), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# --- Visualization Functions ---
fig, ax = None, None

def setup_visualization():
    """
    Initialize matplotlib for interactive mode.
    """
    global fig, ax
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))


def draw_empty_board():
    """
    Draw an NxN checkerboard background with chess coordinates.
    """
    # 1) Checker pattern as before
    pattern = np.fromfunction(lambda i, j: (i + j) % 2, (N, N))
    ax.imshow(pattern, cmap='binary', interpolation='nearest')

    # 2) Major ticks at every square
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))

    # 3) Label files a–h along the x-axis
    files = [chr(ord('a') + i) for i in range(N)]
    ax.set_xticklabels(files)
    ax.xaxis.set_ticks_position('top')      # move labels to the top edge

    # 4) Label ranks 1–N along the y-axis, with "1" at the bottom
    ranks = [str(i + 1) for i in range(N)]
    ax.set_yticklabels(ranks[::-1])         # reverse so '1' is at bottom
    ax.invert_yaxis()                       # makes row-0 plot at the top

    # 5) Draw grid lines on the minor grid (the square boundaries)
    ax.grid(True, which='minor', color='black')


def animate_placement(chromosome, generation, fitness_score, max_score):
    """
    Visualize placing queens one by one for a given chromosome.
    Shows progress within a generation (proof-of-concept).
    """
    ax.clear()
    draw_empty_board()
    # Place each queen sequentially
    for col in range(N):
        row = chromosome[col]
        ax.text(col, row, u"\u265B", fontsize=24,
                ha='center', va='center', color='red')
        ax.set_title(
            f"Gen {generation} | Fit {fitness_score}/{max_score}"
            f"\nPlacing queen {col+1}/{N}")
        fig.canvas.draw(); fig.canvas.flush_events()
        time.sleep(VISUALIZATION_UPDATE_INTERVAL)
    # Short pause after placing all queens
    time.sleep(VISUALIZATION_UPDATE_INTERVAL)




# --- Main Genetic Algorithm ---
def solve_8_queens():
    """
    Run GA from random start positions (proof-of-concept) and animate evolution.
    """
    max_score = N * (N - 1) // 2
    # Initialize population with random chromosomes
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_chromo, best_score, best_gen = None, -1, -1

    setup_visualization()
    for generation in range(MAX_GENERATIONS):
        # Evaluate fitness for each individual
        fitness_scores = [calculate_fitness(ch) for ch in population]
        # Identify current generation's best
        curr_best_score = max(fitness_scores)
        curr_best_chromo = population[fitness_scores.index(curr_best_score)]
        # Update overall best
        if curr_best_score > best_score:
            best_score, best_chromo, best_gen = (
                curr_best_score, curr_best_chromo.copy(), generation)

        # Animate placing queens for this generation's best
        animate_placement(curr_best_chromo, generation,
                          curr_best_score, max_score)

        # Check for perfect solution
        if best_score == max_score:
            print(f"Solution found at generation {best_gen}")
            break

        # Create next generation
        new_pop = [curr_best_chromo.copy()]  # elitism: carry best forward
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitness_scores)
            p2 = tournament_selection(population, fitness_scores)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(mutate(c2))
        population = new_pop

    #Stop animation and show final placement
    plt.ioff(); plt.show()
    return best_chromo, best_score, best_gen

# --- Execution Entry Point ---
if __name__ == '__main__':
    print("Starting 8-Queens GA with interactive evolution visualization...")
    solve_8_queens()
    print("Done.")

