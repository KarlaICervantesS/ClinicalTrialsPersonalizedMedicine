import csv
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GAConfig:
    """
    Configuration object for the genetic algorithm.
    """
    num_subjects: int
    num_groups: int
    num_p: int
    chromosome_length: int
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int
    elitism: int
    seed: int
    H: np.ndarray
    M: np.ndarray
    h_stem: str = "uploaded_data"


def compute_group_sizes(num_subjects: int, num_groups: int) -> list[int]:
    """
    Split subjects as evenly as possible across groups.
    """
    base = num_subjects // num_groups
    remainder = num_subjects % num_groups
    group_size = [base] * num_groups

    for i in range(num_groups - remainder, num_groups):
        group_size[i] += 1

    return group_size


def moments_matrix(m, n_bin, type="l"):
    """
    Build the moments matrix for the selected model type.

    Parameters
    ----------
    m : int
        Total number of covariates excluding the intercept.
    n_bin : int
        Number of binary/categorical covariates.
    type : str
        "l" for linear model,
        "q" for linear + quadratic,
        "full" for linear + quadratic + interactions.
    """
    if type == "l":
        M = (1 / 3) * np.eye(m + 1)
        M[0, 0] = 1

        if n_bin > 0:
            last_elem = np.arange(m + 1 - n_bin, m + 1)
            M[last_elem, last_elem] = 1

        return M

    elif type == "full" and n_bin == 0:
        p = (m + 1) * (m + 2) // 2
        M = np.eye(p)
        diag_vals = np.concatenate([
            [1],
            np.repeat(1 / 3, m),
            np.repeat(1 / 9, m * (m - 1) // 2),
            np.repeat(1 / 5 - 1 / 9, m),
        ])
        np.fill_diagonal(M, diag_vals)

        last_elem = np.arange(p - m, p)
        M[last_elem, last_elem] += 1 / 9
        M[0, last_elem] = 1 / 3
        M[last_elem, 0] = 1 / 3
        return M

    elif type == "q" and n_bin == 0:
        p = 1 + 2 * m
        M = np.eye(p)
        diag_vals = np.concatenate([
            [1],
            np.repeat(1 / 3, m),
            np.repeat(1 / 5 - 1 / 9, m),
        ])
        np.fill_diagonal(M, diag_vals)

        last_elem = np.arange(p - m, p)
        M[last_elem, last_elem] += 1 / 9
        M[0, last_elem] = 1 / 3
        M[last_elem, 0] = 1 / 3
        return M

    raise ValueError("Undefined model")


def fitness(chromosome: List[int], cfg: GAConfig) -> float:
    """
    Compute the objective value for a chromosome.

    Lower is better in the current implementation.
    """
    matrices_hi_hit = []

    # Precompute h_i h_i^T for each subject
    for subject_i in range(cfg.num_subjects):
        hi = cfg.H[subject_i, :].reshape(-1, 1)
        matrices_hi_hit.append(hi @ hi.T)

    group_size = compute_group_sizes(cfg.num_subjects, cfg.num_groups)
    var_k = [0.0] * cfg.num_groups
    big_value = 1e15
    start = 0

    for k, size in enumerate(group_size):
        sum_xhht = np.zeros((cfg.num_p, cfg.num_p))

        for i in range(start, start + size):
            subject_i = chromosome[i]
            sum_xhht += matrices_hi_hit[subject_i]

        start += size

        try:
            inv_sum = np.linalg.inv(sum_xhht)
            var_k[k] = np.trace(inv_sum @ cfg.M)
        except np.linalg.LinAlgError:
            var_k[k] = big_value

    return max(var_k)


def create_chromosome(cfg: GAConfig) -> List[int]:
    """
    Create one chromosome as a random permutation of subject indices.
    """
    return random.sample(range(cfg.num_subjects), cfg.chromosome_length)


def create_population(cfg: GAConfig) -> List[List[int]]:
    """
    Create the initial population.
    """
    return [create_chromosome(cfg) for _ in range(cfg.population_size)]


def tournament_selection(population: List[List[int]], cfg: GAConfig) -> List[int]:
    """
    Select one parent using tournament selection.
    """
    contenders = random.sample(population, cfg.tournament_size)
    return min(contenders, key=lambda ch: fitness(ch, cfg))


def order_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Order crossover (OX) for permutation-based chromosomes.
    """
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    def make_child(pa, pb):
        child = [-1] * n
        child[a:b + 1] = pa[a:b + 1]
        fill_values = [g for g in pb if g not in child]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill_values[idx]
                idx += 1
        return child

    return make_child(p1, p2), make_child(p2, p1)


def swap_mutation(ch: List[int]) -> List[int]:
    """
    Swap two positions in the chromosome.
    """
    c = ch[:]
    i, j = random.sample(range(len(c)), 2)
    c[i], c[j] = c[j], c[i]
    return c


def run_ga(cfg: GAConfig):
    """
    Main genetic algorithm loop.
    Returns:
    - best chromosome
    - best fitness
    - history of best fitness values
    - elapsed execution time
    """
    start = time.perf_counter()

    if cfg.chromosome_length > cfg.num_subjects:
        raise ValueError("chromosome_length must be <= num_subjects")

    random.seed(cfg.seed)
    population = create_population(cfg)
    history_best = []

    for gen in range(cfg.generations):
        # Sort ascending because smaller fitness is better
        population.sort(key=lambda ch: fitness(ch, cfg))
        next_pop = population[: cfg.elitism]

        while len(next_pop) < cfg.population_size:
            p1 = tournament_selection(population, cfg)
            p2 = tournament_selection(population, cfg)

            if random.random() < cfg.crossover_rate:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if random.random() < cfg.mutation_rate:
                c1 = swap_mutation(c1)
            if random.random() < cfg.mutation_rate:
                c2 = swap_mutation(c2)

            next_pop.append(c1)
            if len(next_pop) < cfg.population_size:
                next_pop.append(c2)

        population = next_pop
        best_now = min(population, key=lambda ch: fitness(ch, cfg))
        history_best.append(fitness(best_now, cfg))

    best = min(population, key=lambda ch: fitness(ch, cfg))
    elapsed = time.perf_counter() - start
    return best, fitness(best, cfg), history_best, elapsed


def build_groups_dataframe(best_chromosome, cfg: GAConfig) -> pd.DataFrame:
    """
    Build a table where each column is one group and values are subject IDs.
    """
    chrom = np.asarray(best_chromosome).ravel()
    group_size = compute_group_sizes(cfg.num_subjects, cfg.num_groups)

    starts = [0]
    for s in group_size[:-1]:
        starts.append(starts[-1] + s)

    groups = {
        f"group_{g + 1}": chrom[starts[g]: starts[g] + group_size[g]]
        for g in range(cfg.num_groups)
    }

    return pd.DataFrame(dict((k, pd.Series(v)) for k, v in groups.items()))


def build_assignment_dataframe(best_chromosome, cfg: GAConfig) -> pd.DataFrame:
    """
    Build a two-column table:
    - subject_id
    - assigned group
    """
    group_size = compute_group_sizes(cfg.num_subjects, cfg.num_groups)
    assignment = [None] * len(best_chromosome)

    start = 0
    for g, size in enumerate(group_size):
        for i in range(start, start + size):
            subj = best_chromosome[i]
            assignment[subj] = g + 1
        start += size

    return pd.DataFrame({
        "subject_id": list(range(len(best_chromosome))),
        "group": assignment,
    })


def guardar_asignacion_grupos(chromosome, group_size, cfg):
    """
    Save the final subject-to-group assignment to a CSV file.
    """
    assignment = [None] * len(chromosome)

    start = 0
    for g, size in enumerate(group_size):
        end = start + size
        group_id = g + 1
        for i in range(start, end):
            subj = chromosome[i]
            assignment[subj] = group_id
        start = end

    c = f"{cfg.crossover_rate:.1f}".replace(".", "p")
    m = f"{cfg.mutation_rate:.2f}".replace(".", "p")

    out_file = (
        f"GA_assignment_groups_{cfg.h_stem}_P{cfg.population_size}_G{cfg.generations}"
        f"_C{c}_M{m}_T{cfg.tournament_size}_E{cfg.elitism}.csv"
    )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "group"])
        for subj_id, group in enumerate(assignment):
            writer.writerow([subj_id, group])

    return out_file
