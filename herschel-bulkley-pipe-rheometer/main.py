import argparse
import os
from hb_rheology import analysis, utils

def main():
    """
    Entry point for the Herschel-Bulkley parameter estimation project.
    Provides a command-line interface to run the analysis or generate data.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Estimate Herschel-Bulkley parameters from pipe rheometer data, based on Magnon & Cayeux (2021)."
    )
    parser.add_argument(
        '--reproduce-paper',
        action='store_true',
        help="Run the full analysis pipeline using synthetic data and generate all figures from the paper."
    )
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help="Generate a new synthetic dataset and save it to 'data/synthetic_data.csv'."
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help="Path to a custom input CSV file to run the analysis on."
    )
    args = parser.parse_args()

    if args.reproduce_paper:
        print("="*60)
        print("REPRODUCING FIGURES FROM MAGNON & CAYEUX (2021)")
        print("="*60)
        # Use None as filepath to trigger synthetic data generation within the analysis
        analysis.reproduce_paper_analysis(filepath=None)
    
    elif args.generate_data:
        print("\nGenerating new synthetic dataset...")
        utils.generate_synthetic_data(output_path="data/synthetic_data.csv")
        print("\nDataset saved to 'data/synthetic_data.csv'")

    elif args.input:
        print("="*60)
        print(f"RUNNING ANALYSIS ON CUSTOM DATA: {args.input}")
        print("="*60)
        analysis.reproduce_paper_analysis(filepath=args.input)

    else:
        print("No action specified. Use --help to see available options.")
        print("Example: python main.py --reproduce-paper")

if __name__ == "__main__":
    main()


