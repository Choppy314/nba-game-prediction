"""
Quick script for you to test it
"""
import subprocess
import sys

steps = [
    "src/data_collection.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/evaluate.py"
]

for step in steps:
    print(f"\nRunning: {step}")
    subprocess.run([sys.executable, step], check=True)