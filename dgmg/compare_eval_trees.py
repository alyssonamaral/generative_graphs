# compare_eval_trees.py
# Agrega model_eval.txt das rodadas em árvores e imprime comparação

import os
import glob

def read_eval(path):
    stats = {}
    with open(path, "r") as f:
        for line in f:
            k, v = line.strip().split("\t")
            try:
                stats[k] = float(v)
            except ValueError:
                stats[k] = v
    return stats

def main():
    roots = glob.glob("./runs_trees/train_orders_*")
    if not roots:
        print("Nenhum diretório train_orders_* encontrado em ./runs_trees")
        return
    root = sorted(roots)[-1]
    print(f"Usando: {root}\n")

    rows = []
    for ordname in sorted(os.listdir(root)):
        ord_dir = os.path.join(root, ordname)
        if not os.path.isdir(ord_dir):
            continue
        subdirs = [os.path.join(ord_dir, d) for d in os.listdir(ord_dir)
                   if os.path.isdir(os.path.join(ord_dir, d))]
        if not subdirs:
            continue
        last = sorted(subdirs)[-1]
        eval_path = os.path.join(last, "model_eval.txt")
        if os.path.exists(eval_path):
            s = read_eval(eval_path)
            rows.append((ordname, s))
    if not rows:
        print("Nenhum model_eval.txt encontrado.")
        return

    header = ["ordering", "average_size", "valid_size_ratio", "tree_ratio", "valid_ratio"]
    print("{:12s} {:>12s} {:>18s} {:>10s} {:>12s}".format(*header))
    for ordname, s in rows:
        print("{:12s} {:12.4f} {:18.4f} {:10.4f} {:12.4f}".format(
            ordname,
            s.get("average_size", float("nan")),
            s.get("valid_size_ratio", float("nan")),
            s.get("tree_ratio", float("nan")),
            s.get("valid_ratio", float("nan")),
        ))

if __name__ == "__main__":
    main()
