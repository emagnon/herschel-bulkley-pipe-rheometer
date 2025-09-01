"""
Utility functions for data loading, processing, and synthetic data generation.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from . import model

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 1) -> np.ndarray:
    """
    Applies a Butterworth lowpass filter to the data.
    This is used to simulate sensor smoothing and process raw data.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def generate_synthetic_data(output_path: str, R: float = 0.007875):
    """
    Generates synthetic time-series data mimicking the raw data characteristics
    from the paper for a Carbopol fluid. This version preserves true zero-flow
    samples and adds realistic stress-overshoot bumps after rest-to-flow transitions.

    Args:
        output_path: Path to save the generated CSV file.
        R: Pipe radius (m).
    """
    np.random.seed(2021)  # reproducibility

    # "Ground Truth" parameters from rheometer measurements (paper)
    params_true = {'tau_0': 1.198, 'K': 0.2717, 'n': 0.6389}

    # Time base
    n_points = 2000
    time = np.linspace(0, 200, n_points)

    # Piecewise-constant flow schedule (m^3/s)
    Q = np.zeros(n_points)
    flow_segments = [
        (0, 200, 0), (200, 400, 1.0e-4), (400, 600, 2.0e-4), (600, 800, 4.0e-4),
        (800, 1000, 6.0e-4), (1000, 1200, 8.0e-4), (1200, 1400, 4.0e-4),
        (1400, 1600, 2.0e-4), (1600, 1800, 1.0e-4), (1800, 2000, 0.0)
    ]
    for s, e, q in flow_segments:
        Q[s:e] = q

    # Add small noise, low-pass, then enforce a deadband to keep true zeros
    Q += np.random.normal(0, 5e-6, n_points)
    Q[Q < 0] = 0.0
    Q = butter_lowpass_filter(Q, cutoff=0.4, fs=10, order=2)  # mild sensor smoothing
    Q[np.abs(Q) < 2e-6] = 0.0  # deadband to preserve zeros for Fig.4(b)

    # Ideal pressure gradient from inverse HB model (Eq. 8 + Eq. 5)
    dP_ideal = np.zeros(n_points)
    for i in range(n_points):
        tau_w_i = model.inverse_hb_model(Q[i], **params_true, R=R)
        dP_ideal[i] = 2 * tau_w_i / R  # Pa/m

    # Sensor noise and slow drift
    noise_levels = {'dP1': 20.0, 'dP2': 15.0, 'dP3': 25.0}  # Pa/m std
    drifts = {
        'dP1': np.linspace(0.0, 10.0, n_points),
        'dP2': np.linspace(0.0, -5.0, n_points),
        'dP3': np.linspace(0.0, 8.0, n_points)
    }

    # Identify rest-to-flow edges to inject overshoot bumps (gel-breaking)
    edges = np.where((Q[:-1] == 0.0) & (Q[1:] > 0.0))[0] + 1

    dP_sensors = {}
    for name, noise in noise_levels.items():
        sig = dP_ideal.copy()

        # Add decaying overshoot bumps with magnitudes similar to Fig. 4 (0–1150 Pa/m)
        for e in edges:
            width = 50  # ~5 s at 10 Hz
            A = np.random.uniform(600.0, 1200.0)  # Pa/m
            decay = np.exp(-np.arange(width) / 15.0)
            end = min(e + width, n_points)
            sig[e:end] += A * decay[: end - e]

        # Add noise and drift
        sig += np.random.normal(0.0, noise, n_points)
        sig += drifts[name]

        dP_sensors[name] = sig

    # Add sparse outliers on one sensor to mimic occasional spikes
    outlier_indices = np.random.choice(n_points, size=int(0.01 * n_points), replace=False)
    dP_sensors['dP1'][outlier_indices] *= np.random.uniform(1.5, 2.0)

    # Save
    df = pd.DataFrame({
        'time': time,
        'Q': Q,
        'DP1/L corr': dP_sensors['dP1'],
        'DP2/L corr': dP_sensors['dP2'],
        'DP3/L corr': dP_sensors['dP3']
    })
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated and saved to {output_path}")

def load_and_process_data(filepath: str, R: float, trim_data: bool = True, remove_zero_flow: bool = True, zero_threshold: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data, combines three sensors, applies smoothing, and optionally
    removes near-zero flow points. If filepath is None, synthetic data is generated.

    Args:
        filepath: CSV path or None to generate synthetic data.
        R: Pipe radius (m).
        trim_data: If True, trim extremes after smoothing.
        remove_zero_flow: If True, drop points with Q <= zero_threshold.
        zero_threshold: Threshold to consider as "no flow".
    """
    if filepath is None:
        filepath = "data/synthetic_data.csv"
        if not os.path.exists(filepath):
            print("Synthetic data not found, generating it now...")
            generate_synthetic_data(output_path=filepath, R=R)

    print(f"Loading and processing data from: {filepath}")
    data = pd.read_csv(filepath)

    # Stack all sensor series
    Q = np.concatenate([data['Q'].values, data['Q'].values, data['Q'].values])
    dP_L = np.concatenate([data['DP1/L corr'].values,
                           data['DP2/L corr'].values,
                           data['DP3/L corr'].values])

    # Smooth
    Q_filtered = butter_lowpass_filter(Q, cutoff=0.5, fs=10, order=1)
    dP_L_filtered = butter_lowpass_filter(dP_L, cutoff=0.5, fs=10, order=1)

    # Optionally keep zero/near-zero samples (needed for Fig. 4(b))
    mask = np.ones_like(Q_filtered, dtype=bool)
    if remove_zero_flow:
        mask &= Q_filtered > zero_threshold

    Q_proc = Q_filtered[mask]
    dP_L_proc = dP_L_filtered[mask]

    if trim_data:
        Q_final, dP_L_final = trim_extreme_data(Q_proc, dP_L_proc)
        tau_w_final = model.calculate_wall_shear_stress(dP_L_final, R)
        return Q_final, dP_L_final, tau_w_final
    else:
        tau_w_proc = model.calculate_wall_shear_stress(dP_L_proc, R)
        return Q_proc, dP_L_proc, tau_w_proc

def trim_extreme_data(Q: np.ndarray, dP_L: np.ndarray, trim_fraction: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes a fraction of extreme dP/L values at both ends (default 10%).
    Emulate the paper's strategy of discarding outliers and poorly representative points.
    """
    sort_indices = np.argsort(dP_L)
    num_points = len(sort_indices)
    cut_off = int(trim_fraction * num_points)

    if num_points > 2 * cut_off:
        selected_indices = sort_indices[cut_off:-cut_off]
        Q_final = Q[selected_indices]
        dP_L_final = dP_L[selected_indices]
        print(f"Data trimmed ({trim_fraction*100:.0f}%): {len(Q_final)} points remaining for core analysis.")
    else:
        Q_final, dP_L_final = Q, dP_L
        print("Warning: Not enough data points to perform trimming.")

    return Q_final, dP_L_final

