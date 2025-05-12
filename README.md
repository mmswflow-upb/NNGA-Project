# 8-Queens Solver with Genetic Algorithm

NNGA Project - Year III Sem II

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [Problem Representation](#problem-representation)
  - [Chromosome Encoding](#chromosome-encoding)
  - [Fitness Function](#fitness-function)
  - [Genetic Operators](#genetic-operators)
    - [Selection (Tournament)](#selection-tournament)
    - [Crossover (Order Crossover OX1)](#crossover-order-crossover-ox1)
    - [Mutation (Swap)](#mutation-swap)
    - [Elitism](#elitism)
  - [GA Loop](#ga-loop)
- [Visualization](#visualization)
  - [Interactive Mode](#interactive-mode)
  - [Checkerboard Pattern](#checkerboard-pattern)
  - [Axes, Ticks, and Grid](#axes-ticks-and-grid)
  - [Placing Queens](#placing-queens)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Extending to N-Queens](#extending-to-n-queens)

---

## Introduction

The 8-Queens puzzle asks: can you place eight queens on an 8×8 chessboard so that none attack each other? This implementation uses a Genetic Algorithm—a population-based, evolutionary search—to find a valid placement.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mmswflow-upb/NNGA-Project.git
   ```
2. Install dependencies (I recommend a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

## Installing with Miniconda

If you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, you can quickly create and activate the environment defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate nnproject
```

To apply changes later (after editing `environment.yml`), run:

```bash
conda env update -f environment.yml
```

## Usage

Run the solver:

```bash
python EightQueensProblemSolver.py
```

This will launch an animated Matplotlib window showing queens being placed generation by generation. The script terminates when it finds a solution or reaches the maximum number of generations.

## How It Works

### Problem Representation

- **Search space size**: 8! = 40,320 possible ways to assign one unique row per column.  
- **Goal**: maximize the number of non-attacking queen pairs (28) by eliminating diagonal conflicts.

### Chromosome Encoding

Each individual (chromosome) is a list of length `N` (8) where:

```python
chromosome = [r0, r1, ..., r7]
# index = column, value = row of the queen in that column
```

By using a permutation of `0..N-1`, row- and column-conflicts are automatically avoided.

### Fitness Function

```python
max_pairs = N*(N-1)//2  # total pairs = 28
attacks = count of diagonal conflicts
fitness = max_pairs - attacks
```

- Two queens conflict diagonally if `|i - j| == |row_i - row_j|`.  
- Perfect solution ⇒ `attacks = 0`, `fitness = max_pairs`.

### Genetic Operators

#### Selection (Tournament)

1. Randomly sample `k` individuals from the population.  
2. Choose the one with the highest fitness as a parent.  

This biases reproduction toward fitter individuals while preserving diversity.

#### Crossover (Order Crossover OX1)

1. Pick two cut-points `a < b`.  
2. Child copies parent1’s slice `a..b` in positions `a..b`.  
3. Fill remaining slots by listing genes from parent2 in order, skipping those already copied, wrapping around via modulo.

Produces two valid permutation children.

#### Mutation (Swap)

- With probability `MUTATION_RATE`, pick two indices `i, j` at random and swap `chromosome[i]` ↔ `chromosome[j]`.  
- Keeps the chromosome a permutation, injecting small variability.

#### Elitism

The best individual (champion) in each generation is copied unchanged into the next population, ensuring the best solution never degrades.

### GA Loop

1. **Initialize** population of size `POPULATION_SIZE` with random permutations.  
2. **Evaluate** fitness of each individual.  
3. **Record** and preserve the champion.  
4. **If** champion’s fitness == `max_pairs`, **stop**.  
5. **Build new population**:  
   - Add champion by elitism.  
   - While new population < `POPULATION_SIZE`:  
     1. Select two parents via tournament.  
     2. Crossover to get two children.  
     3. Mutate each child.  
     4. Append to new population.  
6. **Repeat** up to `MAX_GENERATIONS`.

## Visualization

The solver shows an animated view of the best board each generation:

### Interactive Mode

- `plt.ion()` enables non-blocking updates.  
- `fig.canvas.draw()` + `flush_events()` push each frame.  
- `time.sleep(...)` controls the animation speed.

### Checkerboard Pattern

```python
pattern = np.fromfunction(lambda i, j: (i + j) % 2, (N, N))
ax.imshow(pattern, cmap='binary', interpolation='nearest')
```

- Creates an N×N array of 0s/1s for a classic chessboard.  
- `interpolation='nearest'` ensures crisp, non-blurred squares.

### Axes, Ticks, and Grid

```python
ax.set_xticks(np.arange(N))
ax.set_yticks(np.arange(N))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, which='both', color='black')
```

- Places tick marks at each integer to align grid lines with square boundaries.  
- Hides numeric labels for a clean look.  
- Draws black grid lines at every row/column boundary.

### Placing Queens

```python
for col in range(N):
    row = chromosome[col]
    ax.text(col, row, "\u265B", fontsize=24, ha='center', va='center', color='red')
    fig.canvas.draw(); fig.canvas.flush_events(); time.sleep(dt)
```

- Draws the Unicode queen symbol (♛) at each board coordinate.  
- Updates title with generation and fitness info.

## Configuration & Hyperparameters

Located at the top of `main.py`:

- `N` (board size)  
- `POPULATION_SIZE` (e.g., 6 or 50)  
- `MAX_GENERATIONS` (e.g., 500)  
- `MUTATION_RATE` (e.g., 0.1–0.2)  
- `TOURNAMENT_SIZE` (e.g., 5)  
- `VISUALIZATION_UPDATE_INTERVAL` (seconds between queen placements)

Experiment by tuning these values for speed or success rate.

## Extending to N-Queens

Simply change `N = 8` to another integer (e.g., 10). The code adapts:  
- Chromosome length → `N`  
- Fitness max → `N*(N-1)//2`  
- Board drawing → `N×N`
