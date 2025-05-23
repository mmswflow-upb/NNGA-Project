import random
import numpy as np
from plot_utils import init_plot, animate, finalize

# ── GA parameters ───────────────────────────────────────────────────
N         = 8
POP_SIZE  = 10
MAX_GENS  = 500
MUT_RATE  = 0.2
TOUR_SIZE = 2
VIS_DELAY = 0.0008
# ────────────────────────────────────────────────────────────────────

def create_matrix():
    """Generate a random N×N binary matrix with exactly N ones (queens)."""
    mat = np.zeros((N, N), dtype=int)
    for r, c in random.sample([(i, j) for i in range(N) for j in range(N)], N):
        mat[r, c] = 1
    return mat

def fitness(mat):
    """Compute fitness = max non-attacking pairs minus actual attacks."""
    max_pairs = N * (N - 1) // 2
    queens = [(r, c) for r, c in zip(*np.where(mat == 1))]
    attacks = 0
    for i, (r1, c1) in enumerate(queens):
        for r2, c2 in queens[i+1:]:
            if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                attacks += 1
    return max_pairs - attacks

def tour_select(pop, fits):
    """Tournament selection: pick the best of TOUR_SIZE random individuals."""
    contestants = random.sample(list(zip(pop, fits)), TOUR_SIZE)
    return max(contestants, key=lambda x: x[1])[0]

def queen_set_crossover(p1, p2):
    """Set-based crossover preserving exactly N queens without repair."""
    q1 = [(r, c) for r, c in zip(*np.where(p1 == 1))]
    q2 = [(r, c) for r, c in zip(*np.where(p2 == 1))]
    k = random.randint(1, N - 1)
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

def mutate_and_hill_climb(mat):
    """Apply swap-mutation then a 1-step hill climb on the worst queen."""
    # --- mutation: swap one queen with one empty cell ---
    if random.random() < MUT_RATE:
        ones  = list(zip(*np.where(mat == 1)))
        zeros = list(zip(*np.where(mat == 0)))
        r1, c1 = random.choice(ones)
        r0, c0 = random.choice(zeros)
        mat[r1, c1], mat[r0, c0] = 0, 1

    # --- identify most-conflicted queen ---
    queens = [(r, c) for r, c in zip(*np.where(mat == 1))]
    conflict_counts = []
    for (r1, c1) in queens:
        cnt = sum(
            1 for (r2, c2) in queens
            if (r1, c1) != (r2, c2) and
               (r1 == r2 or c1 == c2 or abs(r1-r2) == abs(c1-c2))
        )
        conflict_counts.append(((r1, c1), cnt))

    (worst_r, worst_c), _ = max(conflict_counts, key=lambda x: x[1])

    # --- try moving that queen to any empty cell ---
    zeros = list(zip(*np.where(mat == 0)))
    best_mat = mat.copy()
    best_fit = fitness(mat)
    for r0, c0 in zeros:
        # move queen
        mat[worst_r, worst_c] = 0
        mat[r0, c0]         = 1
        f = fitness(mat)
        if f > best_fit:
            best_fit = f
            best_mat = mat.copy()
        # revert
        mat[r0, c0]         = 0
        mat[worst_r, worst_c] = 1

    return best_mat

def solve_8_queens_matrix():
    max_fit = N * (N - 1) // 2

    # initialize visualization
    init_plot(board_size=N, vis_delay=VIS_DELAY)

    # initial population
    pop = [create_matrix() for _ in range(POP_SIZE)]
    best, best_fit, best_gen = None, -1, -1

    for gen in range(MAX_GENS):
        # evaluate fitness
        fits = [fitness(m) for m in pop]
        champ = pop[fits.index(max(fits))]

        # track global best
        curr_fit = max(fits)
        if curr_fit > best_fit:
            best, best_fit, best_gen = champ.copy(), curr_fit, gen

        # animate current champion
        if not animate(champ, gen, curr_fit, max_fit):
            break

        if best_fit == max_fit:
            print(f"Solution found at generation {best_gen}")
            break

        # build next generation with elitism
        new_pop = [champ.copy()]
        while len(new_pop) < POP_SIZE:
            p1 = tour_select(pop, fits)
            p2 = tour_select(pop, fits)
            child = queen_set_crossover(p1, p2)
            child = mutate_and_hill_climb(child)
            new_pop.append(child)

        pop = new_pop

    # final display
    if animate(best, best_gen, best_fit, max_fit):
        finalize()

solve_8_queens_matrix()
    
