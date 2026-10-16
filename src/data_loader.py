"""Data loading and verification module for renewable energy time-series datasets.

Supports loading default solar, wind, and combined datasets, as well as arbitrary
user-provided CSV files. Handles timestamp parsing, chronological sorting,
column verification, and automatic generation of missing raw data.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

import config
from src.data_generator import create_all_raw_datasets


def resolve_dataset_source(source: Union[str, Path]) -> Tuple[Path, str]:
    """Resolve a source string or Path to an absolute filesystem Path and default target column.

    Parameters
    ----------
    source : Union[str, Path]
        Can be a shorthand keyword ('solar', 'wind', 'combined') or a filesystem path.

    Returns
    -------
    Tuple[Path, str]
        (resolved_file_path, default_target_column)
    """
    source_str = str(source).strip().lower()

    if source_str in ["solar", "solar_energy", "solar_energy.csv"]:
        return config.SOLAR_RAW_PATH, "power_kw"
    elif source_str in ["wind", "wind_energy", "wind_energy.csv"]:
        return config.WIND_RAW_PATH, "power_kw"
    elif source_str in ["combined", "combined_renewable", "combined_renewable_energy.csv"]:
        return config.COMBINED_RAW_PATH, "total_power_kw"
    else:
        path = Path(source)
        if not path.is_absolute():
            path = config.PROJECT_ROOT / path
        return path, "power_kw"


def load_dataset(
    source: Union[str, Path] = "solar",
    target_col: Optional[str] = None,
    required_cols: Optional[List[str]] = None,
    auto_generate: bool = True,
) -> Tuple[pd.DataFrame, Path, str]:
    """Load, validate, parse, and chronologically sort a renewable energy dataset.

    Parameters
    ----------
    source : Union[str, Path]
        Source identifier ('solar', 'wind', 'combined') or file path.
    target_col : Optional[str]
        Target column for power forecasting. If None, inferred based on source.
    required_cols : Optional[List[str]]
        List of expected columns to validate.
    auto_generate : bool
        If True and standard raw files do not exist, automatically generates them.

    Returns
    -------
    Tuple[pd.DataFrame, Path, str]
        (loaded_df, file_path, effective_target_col)
    """
    file_path, default_target = resolve_dataset_source(source)
    effective_target = target_col if target_col is not None else default_target

    # Automatic generation check for missing raw files
    if not file_path.exists():
        if auto_generate and file_path in [
            config.SOLAR_RAW_PATH,
            config.WIND_RAW_PATH,
            config.COMBINED_RAW_PATH,
        ]:
            print(f"[DataLoader] Raw file {file_path.name} not found. Generating default datasets...")
            create_all_raw_datasets(output_dir=config.DATA_RAW_DIR, print_summary=False)
        else:
            raise FileNotFoundError(
                f"[DataLoader] Dataset file not found at: {file_path}. "
                "Please verify the path or provide a valid source keyword ('solar' or 'wind')."
            )

    df = pd.read_csv(file_path)

    # 1. Timestamp validation & parsing
    if "timestamp" not in df.columns:
        # Check if first column is datetime-like
        first_col = df.columns[0]
        try:
            df["timestamp"] = pd.to_datetime(df[first_col])
        except Exception as e:
            raise ValueError(
                f"[DataLoader] 'timestamp' column missing in {file_path.name} and could not convert '{first_col}'."
            ) from e
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Chronological sorting
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 3. Deduplicate exact timestamps
    initial_len = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[DataLoader] Removed {initial_len - len(df)} duplicate timestamp entries.")

    # 4. Target column check
    if effective_target not in df.columns:
        raise ValueError(
            f"[DataLoader] Target column '{effective_target}' not found in {file_path.name}. "
            f"Available columns are: {list(df.columns)}"
        )

    # 5. Required columns validation
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"[DataLoader] Dataset {file_path.name} is missing mandatory columns: {missing}"
            )

    return df, file_path, effective_target


if __name__ == "__main__":
    df_sol, path_sol, target_sol = load_dataset("solar")
    print(f"Loaded {path_sol.name}: {len(df_sol)} rows, target: '{target_sol}'")
    df_wnd, path_wnd, target_wnd = load_dataset("wind")
    print(f"Loaded {path_wnd.name}: {len(df_wnd)} rows, target: '{target_wnd}'")
