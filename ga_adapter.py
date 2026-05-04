from pathlib import Path

import numpy as np
import pandas as pd

from Genetic_Algorithm import (
    GAConfig,
    build_assignment_dataframe,
    build_groups_dataframe,
    compute_group_sizes,
    guardar_asignacion_grupos,
    moments_matrix,
    run_ga,
)


def run_genetic_algorithm(inputs):
    """
    Bridge between the Streamlit app and the genetic algorithm module.

    Responsibilities:
    1. Read the uploaded table already loaded by Streamlit
    2. Build H from the uploaded file
    3. Build M using moments_matrix(...)
    4. Create GAConfig
    5. Run the genetic algorithm
    6. Return results in a format that Streamlit can display
    """
    df = inputs.dataframe.copy()

    total_covariates = (
        inputs.n_quantitative_covariates + inputs.n_categorical_covariates
    )
    n_binary = inputs.n_categorical_covariates

    if total_covariates <= 0:
        raise ValueError("The total number of covariates must be greater than 0.")

    if len(df) < inputs.n_subjects:
        raise ValueError("Not enough rows in the uploaded file for the requested number of subjects.")

    # Assumption:
    # first column = ID
    # next columns = covariates used in the model
    if len(df.columns) < total_covariates + 1:
        raise ValueError(
            "This app assumes the first column is an ID column and the next columns "
            "are covariates. The uploaded file does not have enough columns."
        )

    # Keep only the number of requested subjects
    analysis_df = df.iloc[: inputs.n_subjects].copy()

    # Take the first 'total_covariates' covariate columns after the ID column
    covariates_df = analysis_df.iloc[:, 1 : 1 + total_covariates].copy()

    # Convert non-numeric categorical columns into numeric codes
    # so they can be used in H
    for col in covariates_df.columns:
        if not pd.api.types.is_numeric_dtype(covariates_df[col]):
            covariates_df[col] = pd.Categorical(covariates_df[col]).codes

    # Numeric covariate matrix without intercept
    x = covariates_df.to_numpy(dtype=float)

    # Add intercept column so H has dimension (n_subjects, m + 1)
    intercept = np.ones((x.shape[0], 1))
    H = np.hstack([intercept, x])
    print(H)

    # Build the moments matrix using the requested number of total and binary covariates
    M = moments_matrix(m=total_covariates, n_bin=n_binary, type="l")
    print(M)

    # Sanity check: H and M must have compatible dimensions
    if H.shape[1] != M.shape[0]:
        raise ValueError(
            f"Dimension mismatch: H has {H.shape[1]} columns while M has shape {M.shape}."
        )

    # Build the GA configuration object expected by Genetic_Algorithm.py
    cfg = GAConfig(
        num_subjects=inputs.n_subjects,
        num_groups=inputs.n_groups,
        num_p=H.shape[1],
        chromosome_length=inputs.n_subjects,
        population_size=100,
        generations=200,
        crossover_rate=1.0,
        mutation_rate=0.14,
        tournament_size=2,
        elitism=3,
        seed=123,
        H=H,
        M=M,
        h_stem=inputs.file_stem or "uploaded_file",
    )

    # Run the genetic algorithm
    best_chromosome, best_fit, history, elapsed = run_ga(cfg)

    # Build output tables for display in Streamlit
    group_size = compute_group_sizes(cfg.num_subjects, cfg.num_groups)
    groups_df = build_groups_dataframe(best_chromosome, cfg)
    assignment_df = build_assignment_dataframe(best_chromosome, cfg)

    # Optional: write a CSV output file to disk
    output_csv = guardar_asignacion_grupos(best_chromosome, group_size, cfg)

    return {
        "summary": {
            "elapsed_seconds": elapsed,
            "best_fitness": best_fit,
            "num_subjects": cfg.num_subjects,
            "num_groups": cfg.num_groups,
            "num_covariates": total_covariates,
            "num_binary_covariates": n_binary,
            "num_p": cfg.num_p,
            "output_csv": str(Path(output_csv).resolve()),
        },
        "assignment": assignment_df,
        "groups_table": groups_df,
        "best_score": best_fit,
        "raw_result": {
            "best_chromosome": best_chromosome,
            "history": history,
            "group_size": group_size,
        },
    }
