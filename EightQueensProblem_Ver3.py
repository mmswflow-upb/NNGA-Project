import random, time
import numpy as np
from plot_utils import init_plot, animate, finalize

# ── GA parameters ───────────────────────────────────────────────────
N            = 8
POP_SIZE     = 200
MAX_GENS     = 500
MUT_RATE     = 0.7
TOUR_SIZE    = 20
VIS_DELAY    = 0.0008
# ────────────────────────────────────────────────────────────────────

# ── Chromosome: any 8 queens anywhere ───────────────────────────────
def create_matrix():
    mat = np.zeros((N, N), dtype=int)
    for r, c in random.sample([(i, j) for i in range(N) for j in range(N)], N):
        mat[r, c] = 1
    return mat

# ── Fitness: penalize row + column + diagonal clashes ───────────────
def fitness(mat):
    max_pairs = N * (N - 1) // 2
    queens = [(r, c) for r, c in zip(*np.where(mat == 1))]
    attacks = 0
    for i, (r1, c1) in enumerate(queens):
        for r2, c2 in queens[i+1:]:
            if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                attacks += 1
    return max_pairs - attacks

# ── Selection, crossover, mutation ──────────────────────────────────
def tour_select(pop, fits):
    return max(random.sample(list(zip(pop, fits)), TOUR_SIZE), key=lambda p: p[1])[0]

def queen_set_crossover(p1, p2):
    q1 = [(r, c) for r, c in zip(*np.where(p1 == 1))]
    q2 = [(r, c) for r, c in zip(*np.where(p2 == 1))]
    k = random.randint(1, N-1)
    child_q = random.sample(q1, k)
    for pos in q2:
        if len(child_q) == N:
            break
        if pos not in child_q:
            child_q.append(pos)
    while len(child_q) < N:
        r, c = random.randrange(N), random.randrange(N)
        if (r, c) not in child_q:
            child_q.append((r, c))
    mat = np.zeros((N, N), dtype=int)
    for r, c in child_q:
        mat[r, c] = 1
    return mat


def mutate(mat):
    if random.random() < MUT_RATE:
        ones  = list(zip(*np.where(mat == 1)))
        zeros = list(zip(*np.where(mat == 0)))
        r1, c1 = random.choice(ones)
        r0, c0 = random.choice(zeros)
        mat[r1, c1] = 0
        mat[r0, c0] = 1
    return mat

# ── Main GA Loop with external plotting ─────────────────────────────
def solve_8_queens_matrix():
    max_fit = N * (N - 1) // 2

    # Initialize the plotting window
    init_plot(board_size=N, vis_delay=VIS_DELAY)

    pop = [create_matrix() for _ in range(POP_SIZE)]
    best, best_fit, best_gen = None, -1, -1

    for gen in range(MAX_GENS):
        # Evaluate fitness
        fits = [fitness(m) for m in pop]
        champ = pop[fits.index(max(fits))]

        # Track global best
        curr_fit = max(fits)
        if curr_fit > best_fit:
            best, best_fit, best_gen = champ.copy(), curr_fit, gen

        # Animate this generation's champion
        if not animate(champ, gen, curr_fit, max_fit):
            # User closed the window
            break

        if best_fit == max_fit:
            print(f"Solution found at generation {best_gen}")
            break

        # Build next generation (elitism + tournaments)
        new_pop = [champ.copy()]
        while len(new_pop) < POP_SIZE:
            p1 = tour_select(pop, fits)
            p2 = tour_select(pop, fits)
            c1 = mutate(queen_set_crossover(p1, p2))
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                c2 = mutate(queen_set_crossover(p2, p1))
                new_pop.append(c2)

        pop = new_pop

    # Final display of best solution (if window still open)
    if animate(best, best_gen, best_fit, max_fit):
        finalize()

solve_8_queens_matrix()
