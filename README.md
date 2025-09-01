# Herschel-Bulkley Parameter Estimation from Pipe Rheometer Data

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-GNUGPLv3-green.svg)](https://opensource.org/licenses/GNUGPLv3)

This repository contains a Python implementation of the methodology described in the paper **"Precise Method to Estimate the Herschel-Bulkley Parameters from Pipe Rheometer Measurements"** (Magnon & Cayeux, *Fluids*, 2021).

The goal of this project is to accurately determine the three Herschel-Bulkley (H-B) rheological parameters (`τ₀`, `K`, `n`) for a non-Newtonian fluid using only pipe rheometer measurements of volumetric flow rate (`Q`) and pressure gradient (`dP/dL`).

## Project Motivation and Synthetic Data

The original experimental data used for the publication is proprietary and not available for public distribution. For testing and validation purpose, I added a **synthetic data generator** to the repository.

This generator, located in `hb_rheology/utils.py`, creates a realistic time-series dataset that mimics the behavior of the Carbopol fluid described in the paper. It uses the ground truth parameters obtained from a scientific rheometer as a basis and introduces realistic sensor noise, drift, and outliers.

## Core Methodology

The parameter estimation process follows the physics-based approach from the paper to ensure accuracy, especially at low flow rates. The key steps are:

1.  **Initial Parameter Estimation:** The script first makes initial guesses for `τ₀`, `K`, and `n` by analyzing the physical behavior of the data at high and low shear rates, as shown in Figures 3, 4, and 5 of the paper. This is implemented in `hb_rheology/estimation.py`.

2.  **Physics-Based WRM Correction:** A key contribution of the paper is an analytical formula (Equation 12) for the Weissenberg-Rabinowitsch-Mooney (WRM) correction. This avoids the inaccuracies of standard polynomial fitting.

3.  **Mullineux Optimization:** The corrected shear rate and shear stress values are then used with the Mullineux optimization method (Equation 4) to find the globally optimal flow behavior index `n`, from which `τ₀` and `K` are determined via linear regression.

4.  **Validation:** The results from this method are compared against both the ground-truth rheometer values and a standard Levenberg-Marquardt (L-M) curve fit.

## Repository Structure

```
herschel-bulkley-pipe-rheometer/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── main.py
│
├── hb_rheology/
│   ├── __init__.py
│   ├── analysis.py         # Functions to reproduce paper figures.
│   ├── estimation.py       # Parameter estimation algorithms.
│   ├── model.py            # Herschel-Bulkley physics equations.
│   └── utils.py            # Synthetic data generator.
│
├── data/
└── results/
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/emagnon/herschel-bulkley-pipe-rheometer.git
    cd herschel-bulkley-pipe-rheometer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary way to run this project is to reproduce the full analysis from the paper using the generated synthetic data.

Execute the main script with the `--reproduce-paper` flag:
```bash
python main.py --reproduce-paper
```

This command will:
1.  Generate a synthetic dataset based on the paper's reference parameters.
2.  Run the complete analysis pipeline, including initial estimation and final optimization.
3.  Create and save reproductions of Figures 3-7 from the paper into the `results/` directory .
4.  Print a final comparison table of the estimated parameters against the reference values to the console.

## Expected Results

After running the main script, the `results/` folder will contain the following plots:

*   `figure_3_reproduction.png`: Estimation of `n` from high shear rate data.
*   `figure_4_reproduction.png`: Estimation of `τ₀` from low flow rate data.
*   `figure_5_reproduction.png`: Estimation of `K` using random sampling.
*   `figure_6_reproduction.png`: Comparison of measured vs. calculated pressure losses.
*   `figure_7_reproduction.png`: Comparison of physics-based vs. polynomial derivative estimation.

The console will display example:

```
============================================================
ANALYSIS COMPLETE
============================================================

Final Parameters Comparison:
----------------------------------------
Parameter | Rheometer | L-M (calc) | Mullineux
----------------------------------------
τ₀ (Pa)   |     1.198 |      1.052 |     0.915
K (Pa·sⁿ) |    0.2717 |     0.2650 |    0.2735
n (-)     |    0.6389 |     0.6481 |    0.6412
----------------------------------------

Error Analysis (vs Rheometer):
----------------------------------------
L-M Errors:      τ₀:  12.2%, K:   2.5%, n:   1.4%
Mullineux Errors: τ₀:  23.6%, K:   0.7%, n:   0.4%
----------------------------------------
```

## Citation

If you use this methodology, please cite the original publication:

> Magnon, E.; Cayeux, E. Precise Method to Estimate the Herschel-Bulkley Parameters from Pipe Rheometer Measurements. *Fluids* **2021**, *6*, 157. https://doi.org/10.3390/fluids6040157

## License

This project is licensed under the GNU GPLv3 License.
