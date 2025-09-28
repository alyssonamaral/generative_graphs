# train_by_order.py
# Treina/avalia um modelo por ordenação usando main.py via subprocess

import os
import sys
import argparse
import subprocess
from datetime import datetime

def run(cmd: list):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-dir", type=str, default="./runs/datasets")
    p.add_argument("--orderings", type=str, default="degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--nepochs", type=int, default=25)
    p.add_argument("--logroot", type=str, default="./runs")
    args = p.parse_args()

    orders = [s.strip() for s in args.orderings.split(",") if s.strip()]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_logdir = os.path.join(args.logroot, f"train_orders_{timestamp}")
    os.makedirs(base_logdir, exist_ok=True)

    for ordname in orders:
        dataset_path = os.path.join(args.datasets-dir if hasattr(args, "datasets-dir") else args.datasets_dir, f"cycles_{ordname}.p")
        if not os.path.exists(dataset_path):
            print(f"[!] dataset não encontrado: {dataset_path} — pulei")
            continue

        # cada ordenação com seu subdir
        logdir = os.path.join(base_logdir, ordname)
        os.makedirs(logdir, exist_ok=True)

        cmd = [
            sys.executable, "main.py",
            "--dataset", "cycles",
            "--path-to-dataset", dataset_path,
            "--batch-size", str(args.batch_size),
            "--log-dir", logdir,
            "--nepochs", str(args.nepochs),
        ]
        run(cmd)

    print(f"\nTreinos finalizados. Logs em: {base_logdir}")

if __name__ == "__main__":
    main()
