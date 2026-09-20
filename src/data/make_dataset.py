import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sensor.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_data.npy"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


def load_data():
    """Load the raw pump sensor dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df):
    """
    Clean the raw sensor data and retain only NORMAL operating data.

    Returns:
        normal_data: 2D DataFrame containing only sensor features.
    """

    # Remove columns that are not required for modeling.
    # sensor_15 is completely missing and sensor_50 contains substantial
    # missing values.
    df = df.drop(
        columns=["Unnamed: 0", "timestamp", "sensor_15", "sensor_50"]
    )

    # Separate target attribute from inputs
    status = df["machine_status"]
    features = df.drop(columns=["machine_status"])

    # Handle missing sensor values
    features = features.interpolate(method="linear")
    features = features.bfill()

    # keep only normal operations data for training
    features["machine_status"] = status

    normal_data = features[features["machine_status"] == "NORMAL"].drop(columns=["machine_status"])

    return normal_data


def scale_data(normal_data):
    """
    Fit a MinMaxScaler on the normal training data.

    Returns:
        scaled_data: scaled 2D NumPy array.
        scaler: fitted MinMaxScaler.
    """

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(normal_data)

    return scaled_data, scaler


def save_artifacts(scaled_data, scaler):
    """Save the processed training data and fitted scaler."""

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DATA_PATH, scaled_data)
    joblib.dump(scaler, SCALER_PATH)


def main():
    """Run the complete Phase 1 preprocessing pipeline."""

    print("Loading raw data...")
    df = load_data()

    print(f"Raw data shape: {df.shape}")

    print("Preprocessing data...")
    normal_data = preprocess_data(df)

    print(f"Normal data shape: {normal_data.shape}")

    print("Scaling data...")
    scaled_data, scaler = scale_data(normal_data)

    print(f"Scaled data shape: {scaled_data.shape}")

    print("Saving artifacts...")
    save_artifacts(scaled_data, scaler)

    print("\nValidation:")
    print(f"  Shape: {scaled_data.shape}")
    print(f"  Data type: {scaled_data.dtype}")
    print(f"  Minimum: {scaled_data.min()}")
    print(f"  Maximum: {scaled_data.max()}")
    print(f"  NaNs: {np.isnan(scaled_data).sum()}")
    print(f"  Infs: {np.isinf(scaled_data).sum()}")
    print(f"  Scaler features: {scaler.n_features_in_}")

    print("\nPhase 1 preprocessing completed successfully.")
    print(f"Training data saved to: {PROCESSED_DATA_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()