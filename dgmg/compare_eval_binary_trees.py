# compare_eval_binary_trees.py
import argparse
import glob
import os
import torch

from binary_trees import BinaryTreeModelEvaluation
from model import DGMG
from torch.serialization import add_safe_globals

add_safe_globals([DGMG])

def eval_dir(dirpath, vmin, vmax, nsamples=2000):
    ckpt = os.path.join(dirpath, "model.pth")
    if not os.path.exists(ckpt):
        return None
    model = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.eval()
    ev = BinaryTreeModelEvaluation(v_min=vmin, v_max=vmax, dir=dirpath)
    ev.rollout_and_examine(model, nsamples)
    ev.write_summary()
    return {
        "average_size": ev.average_size,
        "valid_size_ratio": ev.valid_size_ratio,
        "binary_tree_ratio": ev.binary_tree_ratio,
        "valid_ratio": ev.valid_ratio,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-root', type=str, required=False, default='')
    ap.add_argument('--vmin', type=int, default=5)
    ap.add_argument('--vmax', type=int, default=20)
    ap.add_argument('--nsamples', type=int, default=2000)
    args = ap.parse_args()

    # tenta pegar o último diretório train_orders_binary_* se não for passado
    root = args.train_root
    if not root:
        cands = sorted(glob.glob('./runs_bin/train_orders_binary_*'))
        if not cands:
            print("Nada encontrado em ./runs_bin/")
            return
        root = cands[-1]
    print(f"Usando: {root}")

    subdirs = [d for d in sorted(glob.glob(os.path.join(root, '*'))) if os.path.isdir(d)]
    print("\nordering     average_size   valid_size_ratio binary_tree_ratio  valid_ratio")
    for d in subdirs:
        ordering = os.path.basename(d)
        res = eval_dir(d, args.vmin, args.vmax, nsamples=args.nsamples)
        if res is None:
            continue
        print(f"{ordering:<12} {res['average_size']:>12.4f} {res['valid_size_ratio']:>18.4f} {res['binary_tree_ratio']:>18.4f} {res['valid_ratio']:>12.4f}")

if __name__ == '__main__':
    main()
