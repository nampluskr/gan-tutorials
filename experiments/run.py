import subprocess
import sys

def run(file_list):

    for i, file in enumerate(file_list, 1):
        print(f"\n>> [{i}/{len(file_list)}] Running: {file}\n")
        try:
            result = subprocess.run([sys.executable, file], check=True,  encoding="utf-8")
            print(f">> Completed: {file}")
        except subprocess.CalledProcessError:
            print(f"[Error] Script '{file}' failed during execution")
            break
        except FileNotFoundError:
            print(f"[Error] Script '{file}' not found")
            break

if __name__ == "__main__":

    file_list = [
        "./experiments/mnist_infogan_continuous.py",
        "./experiments/mnist_infogan.py",
        "./experiments/mnist_acgan.py",
        "./experiments/mnist_cgan.py",
        "./experiments/mnist_gan.py",
    ]

    run(file_list)