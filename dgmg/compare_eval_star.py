# compare_eval_star.py
import argparse, glob, os, torch
from star import StarModelEvaluation
from model import DGMG

def eval_dir(dirpath, vmin, vmax, nsamples=2000):
    ckpt_pt  = os.path.join(dirpath, "model.pt")
    ckpt_pth = os.path.join(dirpath, "model.pth")  # retrocompatível se existir

    if os.path.exists(ckpt_pt):
        data = torch.load(ckpt_pt, map_location="cpu")
        cfg = data["config"]
        model = DGMG(
            v_max=cfg["v_max"],
            node_hidden_size=cfg["node_hidden_size"],
            num_prop_rounds=cfg["num_prop_rounds"],
        )
        model.load_state_dict(data["state_dict"])
    elif os.path.exists(ckpt_pth):
        from torch.serialization import add_safe_globals
        add_safe_globals([DGMG])
        model = torch.load(ckpt_pth, map_location="cpu", weights_only=False)
    else:
        return None

    model.eval()
    ev = StarModelEvaluation(v_min=vmin, v_max=vmax, dir=dirpath)
    ev.rollout_and_examine(model, nsamples)
    ev.write_summary()
    return {
        "average_size": ev.average_size,
        "valid_size_ratio": ev.valid_size_ratio,
        "star_ratio": ev.star_ratio,
        "valid_ratio": ev.valid_ratio,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-root', type=str, default='')
    ap.add_argument('--vmin', type=int, default=5)
    ap.add_argument('--vmax', type=int, default=20)
    ap.add_argument('--nsamples', type=int, default=2000)
    args = ap.parse_args()

    root = args.train_root
    if not root:
        cands = sorted(glob.glob('./runs_star/train_orders_star_*'))
        if not cands:
            print("Nada encontrado em ./runs_star/")
            return
        root = cands[-1]
    print(f"Usando: {root}")

    subdirs = [d for d in sorted(glob.glob(os.path.join(root, '*'))) if os.path.isdir(d)]
    print("\nordering     average_size   valid_size_ratio star_ratio  valid_ratio")
    for d in subdirs:
        ordering = os.path.basename(d)
        res = eval_dir(d, args.vmin, args.vmax, nsamples=args.nsamples)
        if res is None: 
            continue
        print(f"{ordering:<12} {res['average_size']:>12.4f} {res['valid_size_ratio']:>18.4f} {res['star_ratio']:>10.4f} {res['valid_ratio']:>12.4f}")

if __name__ == '__main__':
    main()
