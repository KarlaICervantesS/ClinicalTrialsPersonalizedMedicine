import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ga_adapter import run_genetic_algorithm


@dataclass
class AppInputs:
    """
    Container for the values collected from the Streamlit interface.
    """
    dataframe: pd.DataFrame
    n_subjects: int
    n_groups: int
    n_quantitative_covariates: int
    n_categorical_covariates: int
    file_stem: str


def load_table(uploaded_file: Any, sheet_name: str | int | None) -> pd.DataFrame:
    """
    Load either a CSV or Excel file into a pandas DataFrame.

    Important behavior:
    - CSV: directly read as a table
    - Excel: if no sheet is provided, read the first sheet only
    """
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        if sheet_name is None or str(sheet_name).strip() == "":
            return pd.read_excel(io.BytesIO(content), sheet_name=0)
        return pd.read_excel(io.BytesIO(content), sheet_name=str(sheet_name).strip())

    raise ValueError("Unsupported file format. Please upload CSV, XLSX, or XLS.")


def validate_inputs(inputs: AppInputs) -> list[str]:
    """
    Validate the user inputs before running the algorithm.
    """
    errors: list[str] = []
    df = inputs.dataframe
    total_covariates = (
        inputs.n_quantitative_covariates + inputs.n_categorical_covariates
    )

    if inputs.n_subjects <= 0:
        errors.append("Number of subjects must be greater than 0.")

    if inputs.n_groups <= 0:
        errors.append("Number of groups must be greater than 0.")

    if inputs.n_quantitative_covariates < 0:
        errors.append("Number of quantitative covariates cannot be negative.")

    if inputs.n_categorical_covariates < 0:
        errors.append("Number of categorical covariates cannot be negative.")

    if total_covariates <= 0:
        errors.append("At least one covariate must be provided.")

    if inputs.n_subjects > len(df):
        errors.append("Number of subjects cannot exceed the number of rows in the file.")

    if inputs.n_groups > inputs.n_subjects:
        errors.append("Number of groups cannot exceed the number of subjects.")

    # Assumption:
    # first column = subject ID
    # next columns = model covariates
    if len(df.columns) < total_covariates + 1:
        errors.append(
            "This app assumes the first column is an ID column and the following "
            "columns are covariates. The uploaded file does not have enough columns."
        )

    return errors


# Streamlit page configuration
st.set_page_config(page_title="Genetic Algorithm Allocation App", layout="wide")

st.title("Genetic Algorithm App for Subject Allocation")
st.write(
    "Upload a CSV or Excel file, provide the study parameters, and run the genetic algorithm."
)

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")

    uploaded_file = st.file_uploader(
        "Data file",
        type=["csv", "xlsx", "xls"],
        help="The app assumes the first column is an ID column and the following columns are covariates.",
    )

    sheet_name = st.text_input(
        "Excel sheet name",
        value="",
        help="Only used for Excel files. Leave empty to use the first sheet.",
    )

    n_subjects = st.number_input("Number of subjects", min_value=1, value=20, step=1)
    n_groups = st.number_input("Number of groups", min_value=1, value=2, step=1)

    n_quantitative_covariates = st.number_input(
        "Number of quantitative covariates", min_value=0, value=2, step=1
    )

    n_categorical_covariates = st.number_input(
        "Number of categorical covariates", min_value=0, value=0, step=1
    )

    run_button = st.button("Run algorithm", type="primary", use_container_width=True)

# Stop early if no file has been uploaded
if uploaded_file is None:
    st.info("Upload a file to begin.")
    st.stop()

# Try loading the uploaded file
try:
    df = load_table(uploaded_file, sheet_name)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

# Preview the uploaded data
st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

# Build the app input object
inputs = AppInputs(
    dataframe=df,
    n_subjects=int(n_subjects),
    n_groups=int(n_groups),
    n_quantitative_covariates=int(n_quantitative_covariates),
    n_categorical_covariates=int(n_categorical_covariates),
    file_stem=Path(uploaded_file.name).stem,
)

# Validate before running
errors = validate_inputs(inputs)
if errors:
    for error in errors:
        st.error(error)
    st.stop()

total_covariates = (
    inputs.n_quantitative_covariates + inputs.n_categorical_covariates
)

# Show a quick summary of the chosen settings
st.subheader("Input Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Subjects")
    st.write(inputs.n_subjects)
with col2:
    st.write("Groups")
    st.write(inputs.n_groups)
with col3:
    st.write("Total covariates")
    st.write(total_covariates)

col4, col5 = st.columns(2)
with col4:
    st.write("Quantitative covariates")
    st.write(inputs.n_quantitative_covariates)
with col5:
    st.write("Categorical covariates")
    st.write(inputs.n_categorical_covariates)

# Run the GA only when the button is pressed
if run_button:
    with st.spinner("Running genetic algorithm..."):
        try:
            result = run_genetic_algorithm(inputs)
        except Exception as exc:
            st.error(f"An error occurred while running the genetic algorithm: {exc}")
            st.stop()

    st.success("Execution completed.")

    if "summary" in result:
        st.subheader("Run Summary")
        st.json(result["summary"])

    if "assignment" in result and isinstance(result["assignment"], pd.DataFrame):
        st.subheader("Assignment by Subject")
        st.dataframe(result["assignment"], use_container_width=True)

        st.download_button(
            "Download assignment CSV",
            data=result["assignment"].to_csv(index=False).encode("utf-8"),
            file_name=f"assignment_{inputs.file_stem}.csv",
            mime="text/csv",
        )

    if "groups_table" in result and isinstance(result["groups_table"], pd.DataFrame):
        st.subheader("Subjects by Group")
        st.dataframe(result["groups_table"], use_container_width=True)

    if "best_score" in result:
        st.metric("Best fitness", result["best_score"])

    if "raw_result" in result:
        st.subheader("Raw Output")
        st.write(result["raw_result"])
