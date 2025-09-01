"""
Implements the full parameter estimation pipeline from Magnon & Cayeux (2021),
including initial estimation, Levenberg-Marquardt fitting, and the final,
more accurate Mullineux optimization on WRM-corrected data.
"""

import numpy as np
from scipy import optimize, stats
from . import model

def _estimate_initial_n(tau_w: np.ndarray, gamma_N_w: np.ndarray) -> float:
    """Estimates initial 'n' from high shear rate data (Fig 3 method)."""
    mask = (gamma_N_w > 1e-9) & (tau_w > 1e-9)
    ln_tau = np.log(tau_w[mask])
    ln_gamma = np.log(gamma_N_w[mask])

    sort_idx = np.argsort(ln_tau)
    ln_tau_sorted = ln_tau[sort_idx]
    ln_gamma_sorted = ln_gamma[sort_idx]

    # Linear regression on last 25% of data (high shear rate)
    quartile_idx = int(0.75 * len(ln_tau_sorted))
    if len(ln_tau_sorted) - quartile_idx < 2: return 0.6 # Not enough points
    
    slope, _, _, _, _ = stats.linregress(
        ln_tau_sorted[quartile_idx:],
        ln_gamma_sorted[quartile_idx:]
    )
    # The paper uses ln(Q) vs ln(tau_w) where slope is 1/n.
    # Here we use ln(gamma) vs ln(tau_w) which is equivalent.
    return 1.0 / slope if slope != 0 else 0.6

def _estimate_initial_tau0(Q: np.ndarray, tau_w: np.ndarray, q_threshold: float = 1e-6) -> float:
    """
    Estimate initial yield stress tau_0 from low-flow data (Fig. 4b logic).

    - Build bins over the lower quartile of tau_w.
    - For each bin, compute probability of flow P(Q > q_threshold | bin).
    - Only keep bins up to the last bin that still contains any non-flowing samples.
    - Compute a weighted average of tau_w with the probabilities as weights.
    """
    if len(Q) == 0 or len(tau_w) == 0:
        return 0.0

    n_bins = 50
    upper = np.percentile(tau_w, 25)
    bins = np.linspace(0.0, upper if upper > 0 else np.max(tau_w), n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    prob_flow = np.zeros(n_bins - 1)
    any_nonflow = np.zeros(n_bins - 1, dtype=bool)

    for i in range(n_bins - 1):
        mask = (tau_w >= bins[i]) & (tau_w < bins[i + 1])
        if np.any(mask):
            total = np.sum(mask)
            flow = np.sum(Q[mask] > q_threshold)
            nonflow = total - flow
            prob_flow[i] = flow / total
            any_nonflow[i] = nonflow > 0

    if not np.any(any_nonflow):
        valid = tau_w[Q > q_threshold]
        return float(np.min(valid) * 0.5) if valid.size else 0.0

    last_nonflow_idx = np.where(any_nonflow)[0].max()
    weights = prob_flow.copy()
    weights[last_nonflow_idx + 1:] = 0.0  # exclude bins beyond stress overshoot

    if np.sum(weights) <= 0:
        valid = tau_w[Q > q_threshold]
        return float(np.min(valid) * 0.5) if valid.size else 0.0

    tau_0_est = float(np.average(centers, weights=weights))
    return max(0.0, tau_0_est)

def _estimate_initial_K(Q: np.ndarray, tau_w: np.ndarray, R: float, n_init: float, tau_0_init: float) -> float:
    """Estimates initial 'K' using random sampling (Fig 5 method)."""
    K_estimates = []
    mask = (Q > 1e-9) & (tau_w > tau_0_init)
    Q_filt, tau_w_filt = Q[mask], tau_w[mask]
    
    if len(Q_filt) == 0: return 0.3

    for i in range(len(Q_filt)):
        # Using Equation (15) from the paper
        try:
            A = tau_w_filt[i] - tau_0_init
            B = (A**2)/(1+3*n_init) + (2*tau_0_init*A)/(1+2*n_init) + (tau_0_init**2)/(1+n_init)
            
            K_numerator = (np.pi * n_init * A**(1+1/n_init) * B)
            K_denominator = (Q_filt[i] * (tau_w_filt[i]/R)**3)
            
            if K_denominator > 1e-9:
                K_val = (K_numerator / K_denominator)**n_init
                if 0 < K_val < 10:
                    K_estimates.append(K_val)
        except (ValueError, OverflowError):
            continue

    return np.median(K_estimates) if K_estimates else 0.3

def levenberg_marquardt_fit(Q, tau_w, R, initial_params):
    """Performs Levenberg-Marquardt curve fitting to get a baseline parameter set."""
    tau_0_init, K_init, n_init = initial_params

    def objective_function(params, Q_data, tau_w_data, R_pipe):
        tau_0, K, n = params
        if tau_0 < 0 or K <= 0 or n <= 0 or n > 2:
            return np.full_like(Q_data, 1e10)
        
        Q_pred = np.array([model.hb_flow_rate(tw, tau_0, K, n, R_pipe) for tw in tau_w_data])
        return Q_data - Q_pred

    try:
        result = optimize.least_squares(
            objective_function,
            [tau_0_init, K_init, n_init],
            args=(Q, tau_w, R),
            bounds=([0, 1e-4, 0.1], [20, 20, 2.0]),
            method='trf'
        )
        tau_0, K, n = result.x
    except Exception as e:
        print(f"L-M optimization failed: {e}. Using initial parameters.")
        tau_0, K, n = initial_params
        
    return {'tau_0': tau_0, 'K': K, 'n': n}

def mullineux_optimization(tau_w, gamma_w):
    """
    Finds optimal HB parameters using the Mullineux method on WRM-corrected data.
    Implements Equation (4) from Magnon & Cayeux (2021).
    """
    mask = (gamma_w > 1e-9) & (tau_w > 0)
    tau_w_filt, gamma_w_filt = tau_w[mask], gamma_w[mask]

    if len(tau_w_filt) < 3:
        print("Warning: Not enough data for Mullineux optimization.")
        return 0, 0.3, 0.6

    def F_n_determinant(n, tau, gamma):
        m = len(gamma)
        gamma_n = gamma ** n
        log_gamma = np.log(gamma)
        
        # Matrix from Equation (4)
        mat = np.array([
            [m,               np.sum(gamma_n),          np.sum(gamma_n * log_gamma)],
            [np.sum(gamma_n), np.sum(gamma_n**2),       np.sum(gamma_n**2 * log_gamma)],
            [np.sum(tau),     np.sum(tau * gamma_n),    np.sum(tau * gamma_n * log_gamma)]
        ])
        return np.linalg.det(mat)

    # Find root for n
    try:
        n_opt = optimize.brentq(F_n_determinant, 0.1, 1.5, args=(tau_w_filt, gamma_w_filt))
    except (ValueError, RuntimeError):
        n_opt = 0.6 

    # With optimal n, find K and tau_0 by linear regression: tau_w = K * gamma_w^n + tau_0
    slope, intercept, _, _, _ = stats.linregress(gamma_w_filt ** n_opt, tau_w_filt)
    K_opt = slope
    tau_0_opt = intercept

    return {'tau_0': max(0, tau_0_opt), 'K': max(1e-4, K_opt), 'n': n_opt}

def run_full_estimation_pipeline(Q, dP_L, R):
    """
    Executes the complete parameter estimation pipeline from the paper.
    Returns dictionaries for Levenberg-Marquardt and Mullineux methods.
    """
    print("\nStep 1: Calculating initial parameter estimates...")
    tau_w = model.calculate_wall_shear_stress(dP_L, R)
    gamma_N_w = (4 * Q) / (np.pi * R**3)

    n_init = _estimate_initial_n(tau_w, gamma_N_w)
    tau_0_init = _estimate_initial_tau0(Q, tau_w, q_threshold=1e-6)
    K_init = _estimate_initial_K(Q, tau_w, R, n_init, tau_0_init)
    initial_params = (tau_0_init, K_init, n_init)
    print(f"   Initial estimates: τ₀={tau_0_init:.3f}, K={K_init:.4f}, n={n_init:.4f}")

    print("\nStep 2: Performing Levenberg-Marquardt fit for baseline...")
    params_lm = levenberg_marquardt_fit(Q, tau_w, R, initial_params)
    print(f"   L-M results:       τ₀={params_lm['tau_0']:.3f}, K={params_lm['K']:.4f}, n={params_lm['n']:.4f}")

    print("\nStep 3: Performing physics-based WRM correction...")
    wrm_deriv = model.calculate_wrm_derivative(tau_w, params_lm['tau_0'], params_lm['n'])

    # Eq. (6): gamma_w = gamma_N_w * (0.75 + 0.25 * dln(gamma_N,w)/dln(tau_w))
    gamma_w_corrected = gamma_N_w * (0.75 + 0.25 * wrm_deriv)

    valid_mask = ~np.isnan(gamma_w_corrected)
    tau_w_for_mullineux = tau_w[valid_mask]
    gamma_w_for_mullineux = gamma_w_corrected[valid_mask]
    print(f"   WRM correction applied, using {len(gamma_w_for_mullineux)} points.")

    print("\nStep 4: Running Mullineux optimization on corrected data...")
    params_mullineux = mullineux_optimization(tau_w_for_mullineux, gamma_w_for_mullineux)
    print(f"   Mullineux results:   τ₀={params_mullineux['tau_0']:.3f}, K={params_mullineux['K']:.4f}, n={params_mullineux['n']:.4f}")

    return params_lm, params_mullineux

