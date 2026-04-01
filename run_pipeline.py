import os
import sys
import subprocess
from pathlib import Path

MODEL_FILES = [
    "decision_tree.pkl",
    "random_forest.pkl",
    "knn.pkl",
    "ridge.pkl",
    "lasso.pkl",
]

def run_step(script_name):
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def models_exist():
    return all(Path(model_file).exists() for model_file in MODEL_FILES)


if __name__ == "__main__":
    print("\nStarting full pipeline...")

    # step 1: train models only if they do not already exist
    if not models_exist():
        print("\nModel pickle files not found. Training models first...")
        run_step("milestone2_models.py")
    else:
        print("\nModel pickle files found. Skipping training step.")

    # step 2: run evaluation + plots + analysis
    run_step("evaluation.py")

    # step 3: run sample demo prediction
    run_step("demo.py")

    print("\nPipeline completed successfully.")