import random
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration & GA Parameters ---
N = 8                  # Board size (8×8)
POPULATION_SIZE = 50   # Number of candidate solutions each generation
MAX_GENERATIONS = 500  # Maximum generations to evolve
MUTATION_RATE = 0.8   # Mutation probability
TOURNAMENT_SIZE = 5    # Individuals in each tournament
VISUALIZATION_UPDATE_INTERVAL = 0.05  # Delay between plot updates

# --- Helper Functions ---
def create_chromosome_matrix():
    """
    Generate a random N×N binary matrix with exactly N ones (queens).
    """
    mat = np.zeros((N, N), dtype=int)
    ones = random.sample(range(N*N), N)
    for idx in ones:
        i, j = divmod(idx, N)
        mat[i, j] = 1
    return mat


def calculate_fitness_matrix(mat):
   
    max_pairs = N * (N - 1) // 2
    queens = [(i, j) for i in range(N) for j in range(N) if mat[i, j] == 1]
    row_conflicts = col_conflicts = diag_conflicts = 0
    for a in range(len(queens)):
        for b in range(a + 1, len(queens)):
            r1, c1 = queens[a]
            r2, c2 = queens[b]
            if r1 == r2:
                row_conflicts += 1
            if c1 == c2:
                col_conflicts += 1
            if abs(r1 - r2) == abs(c1 - c2):
                diag_conflicts += 1
    penalty =  row_conflicts +  col_conflicts + diag_conflicts
    return max_pairs - penalty

# --- Selection: Tournament ---
def tournament_selection(pop, fits):
    contestants = random.sample(list(zip(pop, fits)), TOURNAMENT_SIZE)
    return max(contestants, key=lambda x: x[1])[0]

# --- Conflict-Guided Crossover ---
def conflict_guided_crossover(p1, p2):
    """
    Build two children by inheriting least-conflicting columns first.
    1. Score conflicts per column in each parent.
    2. Copy columns with fewer conflicts up to N//2 queens.
    3. Fill remaining queens from the other parent’s empty columns.
    4. Repair to ensure exactly N queens.
    """
    def col_conflicts(mat):
        conflicts = [0] * N
        queens = [(i, j) for i in range(N) for j in range(N) if mat[i, j] == 1]
        for a in range(len(queens)):
            for b in range(a + 1, len(queens)):
                r1, c1 = queens[a]
                r2, c2 = queens[b]
                if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                    conflicts[c1] += 1
                    conflicts[c2] += 1
        return conflicts

    # compute conflict counts per column
    conf1 = col_conflicts(p1)
    conf2 = col_conflicts(p2)
    # sort columns by minimal conflict
    cols_sorted = sorted(range(N), key=lambda c: min(conf1[c], conf2[c]))

    # initialize children
    c1 = np.zeros((N, N), dtype=int)
    c2 = np.zeros((N, N), dtype=int)
    placed1 = placed2 = 0
    half = N // 2

    # step 1: copy least-conflicting columns up to half queens
    for c in cols_sorted:
        if placed1 < half:
            src = p1 if conf1[c] < conf2[c] else p2
            c1[:, c] = src[:, c]
            placed1 += src[:, c].sum()
        if placed2 < half:
            src = p2 if conf2[c] < conf1[c] else p1
            c2[:, c] = src[:, c]
            placed2 += src[:, c].sum()

    # step 2: fill remaining queens from the other parent
    def fill_remaining(child, src, to_place):
        for c in range(N):
            if to_place <= 0:
                break
            if child[:, c].sum() == 0:
                rows = np.where(src[:, c] == 1)[0]
                if rows.size > 0:
                    child[rows[0], c] = 1
                    to_place -= 1
        return to_place

    rem1 = N - c1.sum()
    rem2 = N - c2.sum()
    rem1 = fill_remaining(c1, p2, rem1)
    rem2 = fill_remaining(c2, p1, rem2)

    # final repair to ensure exactly N queens
    return repair_matrix(c1), repair_matrix(c2)

# --- Repair to fix queen count ---
def repair_matrix(mat):
    total = mat.sum()
    if total > N:
        ones = list(zip(*np.where(mat == 1)))
        for _ in range(total - N):
            i, j = random.choice(ones)
            mat[i, j] = 0
            ones.remove((i, j))
    elif total < N:
        zeros = list(zip(*np.where(mat == 0)))
        for _ in range(N - total):
            i, j = random.choice(zeros)
            mat[i, j] = 1
            zeros.remove((i, j))
    return mat

# --- Mutation: swap two positions ---
def mutate_matrix(mat):
    if random.random() < MUTATION_RATE:
        ones = list(zip(*np.where(mat == 1)))
        zeros = list(zip(*np.where(mat == 0)))
        i1, j1 = random.choice(ones)
        i0, j0 = random.choice(zeros)
        mat[i1, j1] = 0
        mat[i0, j0] = 1
    return mat

# --- Visualization Setup ---
fig, ax = None, None
def setup_visualization():
    global fig, ax
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

def draw_empty_board():
    board = np.fromfunction(lambda i, j: (i + j) % 2, (N, N))
    ax.imshow(board, cmap='binary', interpolation='nearest')
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='black')


def animate_placement_matrix(mat, gen, fit, max_score):
    ax.clear()
    draw_empty_board()
    coords = [(i, j) for i in range(N) for j in range(N) if mat[i, j] == 1]
    for idx, (i, j) in enumerate(coords):
        ax.text(j, i, u"\u265B", fontsize=24, ha='center', va='center', color='red')
        ax.set_title(f"Gen {gen} | Fit {fit}/{max_score}\nPlacing queen {idx+1}/{N}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(VISUALIZATION_UPDATE_INTERVAL)
    time.sleep(VISUALIZATION_UPDATE_INTERVAL)

# --- Main GA Loop using conflict-guided crossover ---
def solve_8_queens_matrix():
    max_score = N * (N - 1) // 2
    pop = [create_chromosome_matrix() for _ in range(POPULATION_SIZE)]
    best, best_score, best_gen = None, -1, -1
    setup_visualization()
    for gen in range(MAX_GENERATIONS):
        fits = [calculate_fitness_matrix(m) for m in pop]
        curr = max(fits)
        idx = fits.index(curr)
        champion = pop[idx].copy()
        if curr > best_score:
            best_score, best, best_gen = curr, champion.copy(), gen
        animate_placement_matrix(champion, gen, curr, max_score)
        if best_score == max_score:
            print(f"Solution found at generation {best_gen}")
            break
        new_pop = [best.copy()]
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            c1, c2 = conflict_guided_crossover(p1, p2)
            new_pop.append(mutate_matrix(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(mutate_matrix(c2))
        pop = new_pop
    animate_placement_matrix(best, best_gen, best_score, max_score)
    plt.ioff()
    plt.show()
    return best, best_score, best_gen

if __name__ == '__main__':
    print("Starting GA 8-Queens with conflict-guided crossover...")
    solve_8_queens_matrix()