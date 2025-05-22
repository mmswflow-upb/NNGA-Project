# 8 Queens Problem Solved with Genetic Algorithms

## Table of Contents

1. [Overview](#overview)  
2. [Implementation](#implementation)  
   2.1 [Version 1: Permutation-Vector GA](#version-1-permutation-vector-ga)  
   2.2 [Version 2: Matrix GA + Repair](#version-2-matrix-ga--repair)  
   2.3 [Version 3: Matrix GA without Repair (Queen-Set Crossover)](#version-3-matrix-ga-without-repair)  
3. [Testing and Results](#testing-and-results)  
4. [Visualization](#visualization)  
5. [Setup Instructions](#setup-instructions)  
6. [Next Steps](#next-steps)  

## Overview

This is a summary of what I've tried and tested (in chronological order) in my journey towards solving the chess-related 8-queens problem.

## Implementation

1. Permutation-Vector GA
2. Matrix GA with Repair
3. Matrix GA without Repair (Queen-Set Crossover)
4. Possibility of Parallelization & N-Queens

## Version 1: Permutation-Vector GA
- *Encoding:* length-8 list, index=column, value=row  
  ```
  Example: [4, 1, 7, 5, 2, 0, 6, 3]  → automatically one queen per column and row
  ```

- *Fitness:* count non-attacking diagonal pairs  
  ```python
  fitness = C(8,2) – number of diagonal conflicts ; (max score = 28)
  ```  
  

- *Operators:*
  - Order Crossover (OX1) to splice permutations, so basically I would keep the same columns of the queens but they'd have different rows
  -  Swap-mutation: swap two elements (I would swap the columns of the queens and keep them on the same rows)


- *Note:* It uses "domain knowledge" to build an optimal genetic algorithm, by making an assumption that queens will never occupy the same columns or rows,
and that they would only attack each other diagonally

## Version 2: Matrix GA + Repair
- *Encoding:* 8×8 binary (0s or 1s) matrix with up to 8 ones  
  (queens can collide in rows OR columns OR diagonals), there cannot be two queens on the same cell
  
- *Fitness:* penalize all clashes (row + column + diagonal), the idea is that only one of the penalties can be applied at a time,
  two queens cannot occupy the same cell, so the maximum penalty per pair remains one.   
  ```python
    fitness = C(8,2) – (row_conflicts + col_conflicts + diag_conflicts) ; max ifitness is still 28
  ```
  
- *Operators:*
  - Conflict-guided crossover: copy “least-conflicting” columns from parents, this caused a problem that I will discuss below...
  - Mutation: swap one 1 with one 0, by randomly selected a queen and an empty cell, and swapping the 1 and 0 in the old, respectively the new cell, thus introducing new configurations that crossovers could never achieve alone
  - Repair: after crossover, randomly add missing ones so sum(matrix)=8, why did this happen? 
    Because in crossovers, I compared the columns of each two parents, even empty columns... this resulted in some generations with less than 8 queens.
   

- *Note:* I didnt want an extra function for filling the boards with the right number of queens, also I didn't like the way I was copying whole columns instead of taking individual queens.

## Version 3: Matrix GA without Repair
- Encoding: same 8×8 binary matrix, but operators always preserve exactly 8 queens

- Queen-Set Crossover:
  1. Extract parent1’s queen-coordinate set (8 pairs)
  2. Randomly sample `k` of those into child
  3. Fill remaining slots with non-duplicate queens from parent2
  4. If still short (due to duplicates), randomly add until 8

- Mutation: move one queen to a random empty cell (like in the previous implementation)

- Fitness: same full penalty of row + column + diagonal conflicts

- *Note:* At this point I also (finally) decided to separate the plotting logic from the actual algorithm and put it in a plot_utils.py

## Testing and Results

## Visualization

- Animates the best candidate each generation using Matplotlib  
- Draws an 8×8 checkerboard, then places eight red ♛ one by one, left-to-right  
- Title shows generation number and fitness score  
- Only the 3rd version uses the module `plot_utils.py` for `init_plot(), animate(), finalize()`, but the other 2 display the chessboards in the same way


## Setup Instructions

1. Clone this repository.
2. Dependencies  

   2.1. (Optional) Create Conda env:  
       ```bash
         conda env create -f environment.yml
         conda activate nnproject
       ```
   
   2.2. Or install via pip:  
       ```bash
         pip install -r requirements.txt
       ```


4. Run the desired variant:

    ```python
     python EightQueensProblem_Ver1.py
     python EightQueensProblem_Ver2.py
     python EightQueensProblem_Ver3.py
    ```

## Next Steps

- **Implement an Island Model**  
  - Partition the population into multiple sub-populations (“islands”), each evolving independently with its own crossover/mutation parameters.  
  - Periodically migrate top individuals between islands to maintain diversity and avoid premature convergence.

- **Parallelize Fitness Evaluation**  
  - Distribute the costly conflict-counting step across multiple CPU cores or worker processes.  
  - Use Python’s multiprocessing or a parallel computing library (e.g., Dask, Ray) to evaluate large populations in parallel, reducing generation time.

- **Leverage GPU for Large-Scale N-Queens**  
  - Port fitness and mutation operators to run on the GPU via CUDA or OpenCL (e.g., with Numba or PyTorch), especially for N≫8.  
  - Batch-process many candidate boards simultaneously to exploit SIMD parallelism.

- **Dynamic Migration Strategies**  
  - Experiment with migration frequency and size: synchronous vs. asynchronous migration, fixed vs. adaptive intervals.  
  - Monitor island diversity and trigger migrations when stagnation is detected.

- **Scale to General N-Queens (N > 8)**  
  - Benchmark performance growth as N increases; adjust GA parameters (population size, mutation rate) for larger problem sizes.  
  - Incorporate problem-specific heuristics (e.g., hill-climbing local search interleaved with GA) to handle the exponentially growing search space.

