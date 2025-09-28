# train_by_order_trees.py
# Treina/avalia um modelo por ordenação (árvores) chamando main.py via subprocess

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
    p.add_argument("--datasets-dir", type=str, default="./runs/datasets_trees")
    p.add_argument("--orderings", type=str,
                   default="degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--nepochs", type=int, default=25)
    p.add_argument("--logroot", type=str, default="./runs_trees")
    # (opcional) alinhar janela de tamanho avaliada
    p.add_argument("--min-size", type=int, default=5)
    p.add_argument("--max-size", type=int, default=20)
    args = p.parse_args()

    orders = [s.strip() for s in args.orderings.split(",") if s.strip()]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_logdir = os.path.join(args.logroot, f"train_orders_{timestamp}")
    os.makedirs(base_logdir, exist_ok=True)

    for ordname in orders:
        dataset_path = os.path.join(args.datasets_dir, f"trees_{ordname}.p")
        if not os.path.exists(dataset_path):
            print(f"[!] dataset não encontrado: {dataset_path} — pulei")
            continue

        logdir = os.path.join(base_logdir, ordname)
        os.makedirs(logdir, exist_ok=True)

        cmd = [
            sys.executable, "main.py",
            "--dataset", "trees",
            "--path-to-dataset", dataset_path,
            "--batch-size", str(args.batch_size),
            "--log-dir", logdir,
            "--nepochs", str(args.nepochs),
            "--min-size", str(args.min_size),
            "--max-size", str(args.max_size),
        ]
        run(cmd)

    print(f"\nTreinos finalizados. Logs em: {base_logdir}")

if __name__ == "__main__":
    main()
