import subprocess
import sys

def run(module: str):
    print(f"\n== Running: python -m {module} ==")
    subprocess.run([sys.executable, "-m", module], check=True)

def main():
    run("src.generate_data")
    run("src.data_prep")
    run("src.train")
    run("src.evaluate")

if __name__ == "__main__":
    main()