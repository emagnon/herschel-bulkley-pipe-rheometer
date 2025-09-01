"""
Contains the core physics-based equations for the Herschel-Bulkley model as described in Magnon & Cayeux (2021).
"""

import numpy as np
from scipy.optimize import fsolve

def calculate_wall_shear_stress(dP_dL: np.ndarray, R: float) -> np.ndarray:
    """
    Calculates wall shear stress from the pressure gradient.
    Implements Equation (5).

    Args:
        dP_dL: Pressure gradient (Pa/m).
        R: Pipe radius (m).

    Returns:
        Wall shear stress (Pa).
    """
    return 0.5 * R * dP_dL

def hb_flow_rate(tau_w: float, tau_0: float, K: float, n: float, R: float) -> float:
    """
    Calculates the volumetric flow rate (Q) for a Herschel-Bulkley fluid.
    Implements Equation (8).

    Args:
        tau_w: Wall shear stress (Pa).
        tau_0: Yield stress (Pa).
        K: Consistency index (Pa·s^n).
        n: Flow behavior index (-).
        R: Pipe radius (m).

    Returns:
        Volumetric flow rate (m³/s).
    """
    if tau_w <= tau_0 or tau_w <= 0:
        return 0.0
    
    A = tau_w - tau_0
    term1 = (A**2) / (1 + 3*n)
    term2 = (2 * tau_0 * A) / (1 + 2*n)
    term3 = (tau_0**2) / (1 + n)
    B = term1 + term2 + term3
    
    # K must be positive
    K_safe = max(K, 1e-9)
    
    Q = (np.pi * n * A**(1 + 1/n)) / (K_safe**(1/n) * (tau_w/R)**3) * B
    return Q

def inverse_hb_model(Q_target: float, tau_0: float, K: float, n: float, R: float) -> float:
    """
    Calculates the wall shear stress (tau_w) for a given flow rate (Q).
    This is the inverse of the hb_flow_rate function, solved numerically.

    Args:
        Q_target: The target volumetric flow rate (m³/s).
        tau_0, K, n, R: Herschel-Bulkley parameters.

    Returns:
        The corresponding wall shear stress (Pa).
    """
    if Q_target <= 1e-10:
        return 0.0

    def equation_to_solve(tau_w_guess: float) -> float:
        """Function to find the root of, where Q_calc - Q_target = 0."""
        Q_calc = hb_flow_rate(tau_w_guess, tau_0, K, n, R)
        return Q_calc - Q_target

    # Provide a good initial guess based on the power-law approximation, improves solver stability
    gamma_N_w_approx = 4 * Q_target / (np.pi * R**3)
    tau_w_initial_guess = tau_0 + K * (gamma_N_w_approx)**n

    try:
        tau_w_solution, _, success, _ = fsolve(equation_to_solve, x0=tau_w_initial_guess, full_output=True)
        if success == 1:
            return tau_w_solution[0]
        else:
            return tau_w_initial_guess
    except (ValueError, RuntimeError):
        return tau_w_initial_guess # Fallback on error

def calculate_wrm_derivative(tau_w: np.ndarray, tau_0: float, n: float) -> np.ndarray:
    """
    Physics-based derivative used in Eq. (6):
    d ln(gamma_N,w) / d ln(tau_w) from Eq. (12) of the paper.

    Returns an array with the same shape as tau_w.
    """
    tau_w = np.asarray(tau_w, dtype=float)
    # Constants from Eq. (10)
    C2 = (1.0 + n) * (1.0 + 2.0 * n)
    C1 = 2.0 * n * tau_0 * (1.0 + n)
    C0 = (6.0 * n - 4.0 * n * n) * (tau_0 ** 2)

    # Avoid division by zero when tau_w ~ tau_0
    eps = 1e-12
    denom1 = np.maximum(np.abs(tau_w - tau_0), eps)
    denom2 = C2 * tau_w**2 + C1 * tau_w + C0
    denom2 = np.where(np.abs(denom2) < eps, np.sign(denom2) * eps + (denom2 == 0) * eps, denom2)

    term1 = ((n + 1.0) / n) * (tau_w / denom1)
    term2 = (-C1 * tau_w + 2.0 * C0) / denom2
    deriv = term1 + term2 - 1.0
    return deriv

