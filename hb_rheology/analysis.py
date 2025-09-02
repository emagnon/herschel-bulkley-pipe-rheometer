"""
Contains functions for reproducing the figures from Magnon & Cayeux (2021) and running the complete analysis pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from . import utils, estimation, model

# --- Helper functions ---

def _last_nonflow_tauw(Q: np.ndarray, tau_w: np.ndarray, q_threshold: float = 1e-6, n_bins: int = 50) -> float:
    """
    Find the upper edge of the last bin (in tau_w) that still contains any non-flow samples (Q <= q_threshold).
    This defines the gel/overshoot band to exclude for polynomial fitting (paper Fig. 4b logic).
    If no such bin exists, returns the minimum tau_w (no exclusion besides non-flow).

    Uses the lower quartile of tau_w to build the histogram, as in the paper.
    """
    if Q.size == 0 or tau_w.size == 0:
        return 0.0

    upper = np.percentile(tau_w, 25)
    upper = float(upper if upper > 0 else np.max(tau_w))
    bins = np.linspace(0.0, upper, n_bins)

    any_nonflow = np.zeros(n_bins - 1, dtype=bool)
    for i in range(n_bins - 1):
        mask = (tau_w >= bins[i]) & (tau_w < bins[i + 1])
        if np.any(mask):
            nonflow = np.sum(Q[mask] <= q_threshold)
            any_nonflow[i] = nonflow > 0

    if not np.any(any_nonflow):
        return float(np.min(tau_w))

    last_idx = np.where(any_nonflow)[0].max()
    # Use the upper edge of that last non-flowing bin
    tau_w_thr = bins[last_idx + 1]
    return float(tau_w_thr)

# --- Figure Reproduction Functions ---

def reproduce_figure_3(tau_w, gamma_N_w):
    """Reproduces Figure 3: Estimation of n from high shear rate data."""
    mask = (gamma_N_w > 1e-9) & (tau_w > 1e-9)
    ln_tau = np.log(tau_w[mask])
    ln_gamma = np.log(gamma_N_w[mask])
    
    sort_idx = np.argsort(ln_tau)
    ln_tau_sorted, ln_gamma_sorted = ln_tau[sort_idx], ln_gamma[sort_idx]
    
    quartile_idx = int(0.75 * len(ln_tau_sorted))
    slope, intercept, _, _, _ = stats.linregress(ln_tau_sorted[quartile_idx:], ln_gamma_sorted[quartile_idx:])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ln_tau_sorted, ln_gamma_sorted, c='red', marker='+', alpha=0.5, label='Measurements (ln scale)')
    x_fit = np.linspace(ln_tau_sorted[quartile_idx], ln_tau_sorted[-1], 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'b-', label=f'Fit (slope={slope:.2f}, n≈{1/slope:.2f})')
    ax.set(xlabel='ln(τw) [Pa]', ylabel='ln(γ_N,w) [1/s]', title='Figure 3 Repro: Initial "n" Estimation')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('results/figure_3_reproduction.png')
    plt.close(fig)

def reproduce_figure_4(Q_raw, tau_w_raw, R, q_threshold: float = 1e-6):
    """
    Reproduces Figure 4: Yield stress estimation from low flow rate data.

    - Part (a): Scatter of Q vs pressure gradient (Pa/m) at low flow;
                uses proper conversion dP/dL = 2*tau_w/R.
    - Part (b): Histogram of non-flowing occurrences and weighted τ0 estimate.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 4 Repro: Initial "τ₀" Estimation', fontsize=14)

    # (a) Low-flow behavior (use radius R, not diameter)
    dP_over_L_raw = 2.0 * tau_w_raw / R
    ax1.scatter(dP_over_L_raw, Q_raw * 1000, alpha=0.3, s=5, label='Measurements')
    ax1.set(xlabel='Pressure Gradient (Pa/m)', ylabel='Flow Rate (L/s)',
            title='(a) Low Flow Rate Behavior', xlim=[0, 1500], ylim=[-0.05, 0.5])
    ax1.axvspan(0, 1150, alpha=0.15, color='gray', label='Gel Breaking Region')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # (b) Histogram of non-flowing probability vs tau_w
    n_bins = 50
    upper = np.percentile(tau_w_raw, 25)
    bins = np.linspace(0.0, upper if upper > 0 else np.max(tau_w_raw), n_bins)
    prob_no_flow = np.zeros(n_bins - 1)

    for i in range(n_bins - 1):
        mask = (tau_w_raw >= bins[i]) & (tau_w_raw < bins[i + 1])
        if np.any(mask):
            total = np.sum(mask)
            nonflow = np.sum(Q_raw[mask] <= q_threshold)
            prob_no_flow[i] = nonflow / total

    tau_0_est = estimation._estimate_initial_tau0(Q_raw, tau_w_raw, q_threshold=q_threshold)
    
    widths = np.diff(bins)
    ax2.bar(bins[:-1], prob_no_flow * 100.0, width=widths, align='edge', label='Prob. of No Flow')
    ax2.axvline(tau_0_est, color='r', ls='--', label=f'Estimated τ₀ = {tau_0_est:.2f} Pa')
    ax2.set(xlabel='Wall Shear Stress τw (Pa)', ylabel='Non-flowing occurrences (%)',
            title='(b) Non-Flowing State Histogram')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig('results/figure_4_reproduction.png')
    plt.close(fig)

def reproduce_figure_6(Q, dP_L, R, params_sets, labels):
    """Reproduces Figure 6: Comparison of measured vs. calculated pressure losses."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(Q * 1000, dP_L, alpha=0.15, s=5, color='gray', label='Measurements')

    # Generate a smooth Q range for plotting model lines
    Q_line = np.linspace(0, Q.max(), 200)
    
    colors = ['blue', 'red', 'green']
    for i, (params, label) in enumerate(zip(params_sets, labels)):
        # Calculate dP_L from Q for each model
        dP_L_model = np.zeros_like(Q_line)
        for j, q_val in enumerate(Q_line):
            tau_w_model = model.inverse_hb_model(q_val, **params, R=R)
            dP_L_model[j] = 2 * tau_w_model / R
        
        ax.plot(Q_line * 1000, dP_L_model, color=colors[i], lw=2, label=label)

    ax.set(xlabel='Flow Rate (L/s)', ylabel='Pressure Gradient (Pa/m)', title='Figure 6 Repro: Model vs. Measurement')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig('results/figure_6_reproduction.png')
    plt.close(fig)

def reproduce_figure_7(tau_w: np.ndarray, gamma_N_w: np.ndarray, params_list: list, labels: list, Q: np.ndarray, q_threshold: float = 1e-6, 
    poly_degrees: tuple = (2, 4), save_path: str = "results/figure_7_reproduction.png"):
    """
    Reproduce Fig. 7 using the paper's methodology.

    - Physics-based curves: d ln(gamma_N,w)/d ln(tau_w) from Eq. (12) evaluated with:
        1) Reference (rheometer) parameters
        2) Levenberg-Marquardt parameters
        3) Mullineux parameters
      Only tau_0 and n are needed for the derivative.

    - Polynomial curves (dashed): least-squares polynomial fit of ln(gamma_N,w) vs ln(tau_w)
      performed on measured data only, with:
        - exclusion of non-flow: Q <= q_threshold
        - exclusion of gel/overshoot band: tau_w <= last_nonflow_tauw
      Fit degrees set by poly_degrees, default (2, 4).
      The derivative shown is the derivative of the fitted polynomial in log–log space.

    Plots derivative vs ln(tau_w).
    """

    # Basic validity mask
    valid = np.isfinite(tau_w) & np.isfinite(gamma_N_w) & np.isfinite(Q)
    tau_w = tau_w[valid]
    gamma_N_w = gamma_N_w[valid]
    Q = Q[valid]

    # Exclude non-flowing first
    flowing_mask = Q > q_threshold
    tau_w_flow = tau_w[flowing_mask]
    gamma_flow = gamma_N_w[flowing_mask]

    # Determine last non-flow tau_w threshold (gel/overshoot upper edge) from the full (including non-flow) data
    tau_w_threshold = _last_nonflow_tauw(Q=Q, tau_w=tau_w, q_threshold=q_threshold, n_bins=50)

    # Mask out the gel/overshoot region for the polynomial fit
    fit_mask = tau_w_flow > tau_w_threshold
    tau_w_fit = tau_w_flow[fit_mask]
    gamma_fit = gamma_flow[fit_mask]

    # Log–log space for fitting
    x_fit = np.log(tau_w_fit)
    y_fit = np.log(gamma_fit)

    # x-grid to draw smooth dashed curves
    x_grid = np.linspace(np.min(x_fit), np.max(x_fit), 400)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Figure 7 Repro: Derivative Estimation Comparison")
    ax.set_xlabel("ln(τw) [Pa]")
    ax.set_ylabel("d ln(γ_N,w)/d ln(τw)")

    # Physics-based curves (scatter) using Eq. (12) for each parameter set
    # Evaluate on the same tau_w domain as used for the polynomial fits (flowing & above threshold)
    tau_w_phys = np.sort(tau_w_fit)
    for params, lab, marker, color in zip(
        params_list,
        labels,
        ['+', 'x', 'o'],
        ['C0', 'C3', 'C2']
    ):
        tau0 = float(params['tau_0'])
        n = float(params['n'])
        deriv = model.calculate_wrm_derivative(tau_w_phys, tau0, n)
        ax.scatter(np.log(tau_w_phys), deriv, s=16, marker=marker, color=color, label=f"Physics-Based ({lab})", alpha=0.85)

    # Polynomial dashed curves (fit on measured ln–ln, derivative is dP/dx of the polynomial in log-space)
    for deg, style, color in zip(poly_degrees, ['--', '--'], ['C1', 'C4']):
        if x_fit.size >= deg + 1:
            coeffs = np.polyfit(x_fit, y_fit, deg=deg)
            p = np.poly1d(coeffs)
            dp = np.polyder(p)
            ax.plot(x_grid, dp(x_grid), linestyle=style, color=color, label=f"Poly Fit (order {deg}) Derivative", linewidth=2.0)

    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# --- Main Analysis Orchestrator ---

def reproduce_paper_analysis(filepath: str, R: float = 0.007875):
    """
    Runs the complete analysis pipeline: loads data, estimates parameters,
    reproduces core figures, and prints a final summary.
    """
    # Load untrimmed data keeping zeros to compute Fig. 4 histogram and thresholds
    Q_raw, dP_L_raw, _ = utils.load_and_process_data(
        filepath, R, trim_data=False, remove_zero_flow=False, zero_threshold=1e-6
    )
    tau_w_raw = model.calculate_wall_shear_stress(dP_L_raw, R)

    print("\nStep: Reproducing Figure 4 (untrimmed, zeros kept)...")
    reproduce_figure_4(Q_raw, tau_w_raw, R, q_threshold=1e-6)

    # For parameter estimation and Figs. 6–7, trim extremes
    Q, dP_L = utils.trim_extreme_data(Q_raw, dP_L_raw)
    tau_w = model.calculate_wall_shear_stress(dP_L, R)
    gamma_N_w = (4.0 * Q) / (np.pi * R**3)

    reproduce_figure_3(tau_w, gamma_N_w)
    
    # Fit pipeline (L-M and Mullineux)
    params_lm, params_mullineux = estimation.run_full_estimation_pipeline(Q, dP_L, R)

    # Reference (rheometer) parameters from the paper (kept for comparison)
    params_rheometer = {'tau_0': 1.198, 'K': 0.2717, 'n': 0.6389}

    # Reproduce Fig. 6 (implementation assumed present)
    reproduce_figure_6(
        Q, dP_L, R,
        [params_rheometer, params_lm, params_mullineux],
        ['Rheometer (Ref)', 'Levenberg-Marquardt', 'Mullineux (Physics-Based)']
    )

    # Reproduce Fig. 7
    reproduce_figure_7(
        tau_w=tau_w,
        gamma_N_w=gamma_N_w,
        params_list=[params_rheometer, params_lm, params_mullineux],
        labels=['Ref', 'L-M', 'Mullineux'],
        Q=Q,
        q_threshold=1e-6,
        poly_degrees=(2, 4),
        save_path="results/figure_7_reproduction.png"
    )

    print("   Figures saved in 'results/'.")
    _print_summary_table(params_rheometer, params_lm, params_mullineux)

def _print_summary_table(p_ref, p_lm, p_mul):
    """Prints the final comparison table and error analysis."""
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE: FINAL PARAMETERS")
    print("="*60)
    
    header = f"{'Parameter':<10} | {'Rheometer (Ref)':>15} | {'L-M (Fit)':>12} | {'Mullineux (Final)':>18}"
    print(header)
    print("-" * len(header))
    print(f"{'τ₀ (Pa)':<10} | {p_ref['tau_0']:15.3f} | {p_lm['tau_0']:12.3f} | {p_mul['tau_0']:18.3f}")
    print(f"{'K (Pa·sⁿ)':<10} | {p_ref['K']:15.4f} | {p_lm['K']:12.4f} | {p_mul['K']:18.4f}")
    print(f"{'n (-)':<10} | {p_ref['n']:15.4f} | {p_lm['n']:12.4f} | {p_mul['n']:18.4f}")
    print("-" * len(header))

    def error(est, ref): return abs(est - ref) / ref * 100 if ref != 0 else 0
    
    print("\nError Analysis (vs. Rheometer Reference):")
    err_lm = (error(p_lm['tau_0'], p_ref['tau_0']), error(p_lm['K'], p_ref['K']), error(p_lm['n'], p_ref['n']))
    err_mul = (error(p_mul['tau_0'], p_ref['tau_0']), error(p_mul['K'], p_ref['K']), error(p_mul['n'], p_ref['n']))
    
    print(f"  L-M Errors:      τ₀: {err_lm[0]:5.1f}%, K: {err_lm[1]:5.1f}%, n: {err_lm[2]:5.1f}%")
    print(f"  Mullineux Errors: τ₀: {err_mul[0]:5.1f}%, K: {err_mul[1]:5.1f}%, n: {err_mul[2]:5.1f}%")
    print("="*60)

