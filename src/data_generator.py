"""Synthetic dataset generator for solar, wind, and combined renewable energy systems.

Generates physically plausible, non-empty time-series datasets based on:
1. Solar PV: Astronomical solar zenith angle modeling, cloud attenuation,
   temperature-derating effects, and inverter clipping.
2. Wind Turbine: Weibull-distributed autoregressive wind speeds, circular wind directions,
   atmospheric barometric variation, and a standard piecewise cubic aerodynamic power curve
   with cut-in, rated, and cut-out limits.
3. Combined System: Synchronized multi-source renewable microgrid generation.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is available in sys.path for direct execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import config



def generate_solar_dataset(
    n_samples: int = config.DEFAULT_SAMPLE_COUNT,
    start_date: str = config.SIMULATION_START_DATE,
    freq: str = config.SAMPLE_FREQUENCY,
    seed: int = config.RANDOM_SEED,
    add_test_anomalies: bool = True,
) -> pd.DataFrame:
    """Generate realistic synthetic solar photovoltaic time-series data.

    Parameters
    ----------
    n_samples : int
        Number of consecutive 15-minute observations (minimum 5,000 required).
    start_date : str
        ISO string for simulation start timestamp.
    freq : str
        Pandas frequency string (e.g. '15min').
    seed : int
        Random seed for reproducibility.
    add_test_anomalies : bool
        If True, injects a controlled fraction (0.3%) of missing values to test
        data cleaning and imputation pipelines.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['timestamp', 'solar_irradiance', 'temperature', 'humidity', 'cloud_cover', 'power_kw']
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    # Time characteristics
    day_of_year = timestamps.dayofyear.values
    hour_of_day = timestamps.hour.values + timestamps.minute.values / 60.0

    # 1. Solar Astronomical Geometry (Latitude ~35.0 N)
    latitude_rad = np.radians(35.0)
    # Solar declination angle delta
    declination = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 81))))
    # Hour angle omega (-180 to +180 deg, 0 at solar noon)
    hour_angle = np.radians((hour_of_day - 12.0) * 15.0)

    # Solar elevation angle alpha: sin(alpha) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(omega)
    sin_alpha = (
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    # Sunlight exists only when elevation > 0
    sun_up = sin_alpha > 0.0

    # Clear-sky Global Horizontal Irradiance (GHI) in W/m²
    # Top-of-atmosphere constant ~1367 W/m², atmospheric transmittance ~0.72
    clear_sky_ghi = np.zeros(n_samples)
    clear_sky_ghi[sun_up] = 1050.0 * np.maximum(0.0, sin_alpha[sun_up]) ** 1.15

    # 2. Cloud Cover Modeling (AR(1) process for synoptic weather fronts)
    cloud_noise = rng.normal(loc=0.0, scale=12.0, size=n_samples)
    cloud_cover = np.zeros(n_samples)
    cloud_cover[0] = 30.0
    for t in range(1, n_samples):
        # Mean-reverting stochastic process around 35% average cloud cover
        cloud_cover[t] = 0.94 * cloud_cover[t - 1] + 0.06 * 35.0 + cloud_noise[t]
    cloud_cover = np.clip(cloud_cover, 0.0, 100.0)

    # 3. Solar Irradiance with Cloud Attenuation
    # Cloud attenuation factor: K = 1 - 0.75 * (cloud / 100)^3
    cloud_factor = 1.0 - 0.75 * (cloud_cover / 100.0) ** 2.5
    irradiance_noise = rng.normal(loc=0.0, scale=10.0, size=n_samples)
    solar_irradiance = clear_sky_ghi * cloud_factor + irradiance_noise
    solar_irradiance = np.where(sun_up, np.maximum(0.0, solar_irradiance), 0.0)

    # 4. Ambient Temperature (°C)
    # Diurnal swing (peaks at 15:00, minimum at 06:00), seasonal warming, cloud dampening
    temp_seasonal = 12.0 + 10.0 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 100)))
    temp_diurnal = 7.0 * np.sin(np.radians((hour_of_day - 9.0) * 15.0))
    temp_cloud_effect = -3.0 * (cloud_cover / 100.0)
    temp_noise = rng.normal(0.0, 0.6, size=n_samples)
    temperature = temp_seasonal + temp_diurnal + temp_cloud_effect + temp_noise

    # 5. Relative Humidity (%)
    # Inversely correlated with temperature and higher on cloudy days
    humidity_base = 72.0 - 1.5 * (temperature - np.mean(temperature)) + 0.25 * cloud_cover
    humidity_noise = rng.normal(0.0, 3.0, size=n_samples)
    humidity = np.clip(humidity_base + humidity_noise, 15.0, 98.0)

    # 6. Solar Photovoltaic Power Output (kW)
    # Cell Temperature estimation: T_cell = T_amb + G * (NOCT - 20) / 800
    # NOCT (Nominal Operating Cell Temperature) ~ 45°C
    t_cell = temperature + solar_irradiance * (45.0 - 20.0) / 800.0
    # Temperature derating: 1 + gamma * (T_cell - 25°C), gamma = -0.004
    temp_derating = 1.0 + config.SOLAR_TEMP_COEFF * (t_cell - 25.0)

    # Power (kW) = Irradiance (kW/m²) * Area (m²) * Efficiency * Derating * Inverter_Eff
    raw_power = (
        (solar_irradiance / 1000.0)
        * config.SOLAR_PANEL_AREA_M2
        * config.SOLAR_BASE_EFFICIENCY
        * np.maximum(0.70, temp_derating)
        * config.SOLAR_INVERTER_EFFICIENCY
    )
    # Add minor sensor/inverter measurement noise (std = 0.2 kW)
    power_noise = rng.normal(0.0, 0.2, size=n_samples)
    power_kw = np.where(solar_irradiance > 5.0, raw_power + power_noise, 0.0)
    power_kw = np.clip(power_kw, 0.0, config.SOLAR_CAPACITY_KW)

    df_solar = pd.DataFrame(
        {
            "timestamp": timestamps,
            "solar_irradiance": np.round(solar_irradiance, 2),
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 1),
            "cloud_cover": np.round(cloud_cover, 1),
            "power_kw": np.round(power_kw, 3),
        }
    )

    # Controlled missing value injection for pipeline validation (0.3%)
    if add_test_anomalies:
        mask_missing = rng.random(size=n_samples) < 0.003
        df_solar.loc[mask_missing, "solar_irradiance"] = np.nan
        df_solar.loc[mask_missing, "power_kw"] = np.nan

    return df_solar


def generate_wind_dataset(
    n_samples: int = config.DEFAULT_SAMPLE_COUNT,
    start_date: str = config.SIMULATION_START_DATE,
    freq: str = config.SAMPLE_FREQUENCY,
    seed: int = config.RANDOM_SEED + 10,
    add_test_anomalies: bool = True,
) -> pd.DataFrame:
    """Generate realistic synthetic wind energy time-series data.

    Parameters
    ----------
    n_samples : int
        Number of consecutive 15-minute observations (minimum 5,000 required).
    start_date : str
        ISO string for simulation start timestamp.
    freq : str
        Pandas frequency string (e.g. '15min').
    seed : int
        Random seed for reproducibility.
    add_test_anomalies : bool
        If True, injects controlled missing values to verify preprocessing robustness.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['timestamp', 'wind_speed', 'wind_direction', 'temperature', 'pressure', 'power_kw']
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    # 1. Wind Speed Simulation (Autoregressive process over Weibull marginals)
    # Weibull scale c = 8.5 m/s, shape k = 2.1 (typical IEC Class II site)
    weibull_base = rng.weibull(a=2.1, size=n_samples) * 8.5
    wind_speed = np.zeros(n_samples)
    wind_speed[0] = weibull_base[0]
    # AR(1) temporal persistence coefficient = 0.92 for turbulent inertia
    for t in range(1, n_samples):
        wind_speed[t] = 0.92 * wind_speed[t - 1] + 0.08 * weibull_base[t]
    # Add gust fluctuations
    gust_noise = rng.normal(0.0, 0.4, size=n_samples)
    wind_speed = np.clip(wind_speed + gust_noise, 0.0, 32.0)

    # 2. Wind Direction Simulation (Circular random walk with prevailing regime)
    direction_drift = rng.normal(loc=0.0, scale=4.0, size=n_samples)
    wind_direction = np.zeros(n_samples)
    wind_direction[0] = 220.0  # Prevailing South-Westerly wind
    for t in range(1, n_samples):
        # Drift with slight restoring pull toward prevailing 220 degrees
        wind_direction[t] = (
            wind_direction[t - 1]
            + direction_drift[t]
            + 0.02 * (220.0 - wind_direction[t - 1])
        ) % 360.0

    # 3. Ambient Temperature (°C)
    day_of_year = timestamps.dayofyear.values
    hour_of_day = timestamps.hour.values + timestamps.minute.values / 60.0
    temp_seasonal = 10.0 + 8.0 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 80)))
    temp_diurnal = 4.0 * np.sin(np.radians((hour_of_day - 10.0) * 15.0))
    temperature = temp_seasonal + temp_diurnal + rng.normal(0.0, 0.5, size=n_samples)

    # 4. Atmospheric Pressure (hPa)
    # Synoptic weather high/low pressure systems with 1013.25 hPa nominal
    pressure_synoptic = rng.normal(0.0, 0.8, size=n_samples)
    pressure = np.zeros(n_samples)
    pressure[0] = 1013.25
    for t in range(1, n_samples):
        pressure[t] = 0.995 * pressure[t - 1] + 0.005 * 1013.25 + pressure_synoptic[t]
    pressure = np.round(pressure, 1)

    # 5. Wind Turbine Aerodynamic Power Curve
    # Parameters from config:
    # v_ci = 3.0 m/s (cut-in)
    # v_r  = 12.0 m/s (rated)
    # v_co = 25.0 m/s (cut-out)
    # P_rated = 500.0 kW
    v_ci = config.WIND_CUT_IN_SPEED
    v_r = config.WIND_RATED_SPEED
    v_co = config.WIND_CUT_OUT_SPEED
    p_rated = config.WIND_TURBINE_RATED_KW

    power_kw = np.zeros(n_samples)

    # Region II (Between cut-in and rated): cubic ramp
    mask_ramp = (wind_speed >= v_ci) & (wind_speed < v_r)
    power_kw[mask_ramp] = p_rated * (
        (wind_speed[mask_ramp] ** 3 - v_ci**3) / (v_r**3 - v_ci**3)
    )

    # Region III (Between rated and cut-out): pitch-controlled constant rated power
    mask_rated = (wind_speed >= v_r) & (wind_speed < v_co)
    power_kw[mask_rated] = p_rated

    # Region IV (Above cut-out): aerodynamic feathering / mechanical braking
    mask_cutout = wind_speed >= v_co
    power_kw[mask_cutout] = 0.0

    # Add small operational turbulence noise (std = 1.5 kW when active)
    active_mask = (wind_speed >= v_ci) & (wind_speed < v_co)
    noise = rng.normal(0.0, 1.5, size=n_samples)
    power_kw[active_mask] = np.clip(power_kw[active_mask] + noise[active_mask], 0.0, p_rated)

    df_wind = pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind_speed": np.round(wind_speed, 2),
            "wind_direction": np.round(wind_direction, 1),
            "temperature": np.round(temperature, 2),
            "pressure": np.round(pressure, 1),
            "power_kw": np.round(power_kw, 3),
        }
    )

    # Controlled missing value injection for cleaning validation (0.3%)
    if add_test_anomalies:
        mask_missing = rng.random(size=n_samples) < 0.003
        df_wind.loc[mask_missing, "wind_speed"] = np.nan
        df_wind.loc[mask_missing, "power_kw"] = np.nan

    return df_wind


def generate_combined_dataset(
    df_solar: Optional[pd.DataFrame] = None,
    df_wind: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine solar and wind datasets into a synchronized multi-source dataset.

    Parameters
    ----------
    df_solar : Optional[pd.DataFrame]
        Pre-generated solar dataframe (or generated fresh if None).
    df_wind : Optional[pd.DataFrame]
        Pre-generated wind dataframe (or generated fresh if None).

    Returns
    -------
    pd.DataFrame
        Combined renewable energy dataframe with solar, wind, and total power columns.
    """
    if df_solar is None:
        df_solar = generate_solar_dataset(add_test_anomalies=False)
    if df_wind is None:
        df_wind = generate_wind_dataset(add_test_anomalies=False)

    df_combined = pd.DataFrame(
        {
            "timestamp": df_solar["timestamp"],
            "solar_irradiance": df_solar["solar_irradiance"],
            "solar_temperature": df_solar["temperature"],
            "solar_humidity": df_solar["humidity"],
            "cloud_cover": df_solar["cloud_cover"],
            "solar_power_kw": df_solar["power_kw"],
            "wind_speed": df_wind["wind_speed"],
            "wind_direction": df_wind["wind_direction"],
            "wind_temperature": df_wind["temperature"],
            "pressure": df_wind["pressure"],
            "wind_power_kw": df_wind["power_kw"],
            "total_power_kw": np.round(
                df_solar["power_kw"].fillna(0) + df_wind["power_kw"].fillna(0), 3
            ),
        }
    )
    return df_combined


def create_all_raw_datasets(
    output_dir: Optional[Path] = None,
    print_summary: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate and persist all required raw CSV datasets to data/raw/.

    Parameters
    ----------
    output_dir : Optional[Path]
        Target folder for CSV outputs. Defaults to config.DATA_RAW_DIR.
    print_summary : bool
        If True, displays rows, columns, date range, and missing count for each dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (solar_df, wind_df, combined_df)
    """
    if output_dir is None:
        output_dir = config.DATA_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    solar_path = output_dir / "solar_energy.csv"
    wind_path = output_dir / "wind_energy.csv"
    combined_path = output_dir / "combined_renewable_energy.csv"

    solar_df = generate_solar_dataset(add_test_anomalies=True)
    wind_df = generate_wind_dataset(add_test_anomalies=True)
    combined_df = generate_combined_dataset(
        solar_df.copy().interpolate(),
        wind_df.copy().interpolate(),
    )

    solar_df.to_csv(solar_path, index=False)
    wind_df.to_csv(wind_path, index=False)
    combined_df.to_csv(combined_path, index=False)

    if print_summary:
        print("=" * 80)
        print("SYNTHETIC DATASET GENERATION SUMMARY (Saved in data/raw/)")
        print("Provenance: Rigorous physically-grounded PV & aerodynamic wind turbine models")
        print("=" * 80)

        for name, df, path in [
            ("Solar Energy Dataset", solar_df, solar_path),
            ("Wind Energy Dataset", wind_df, wind_path),
            ("Combined Renewable Energy Dataset", combined_df, combined_path),
        ]:
            missing_count = int(df.isna().sum().sum())
            start_ts = df["timestamp"].min()
            end_ts = df["timestamp"].max()
            print(f"Dataset      : {name}")
            print(f"File Path    : {path.relative_to(config.PROJECT_ROOT)}")
            print(f"Dimensions   : {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"Date Range   : {start_ts} to {end_ts} (15-min intervals)")
            print(f"Missing Cells: {missing_count} (Controlled for preprocessing test)")
            print(f"Columns      : {list(df.columns)}")
            print("-" * 80)

    return solar_df, wind_df, combined_df


if __name__ == "__main__":
    create_all_raw_datasets(print_summary=True)
