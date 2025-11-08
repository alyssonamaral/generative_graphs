# train_orders_star.py
import argparse, os, time, torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from model import DGMG
from star import StarDataset
from cycles import CyclePrinting
from utils import mkdir_p

def train_one(fname, outdir, max_size, node_hidden_size, num_prop_rounds, lr, nepochs, batch_size):
    ds = StarDataset(fname=fname)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=ds.collate_single)

    model = DGMG(v_max=max_size, node_hidden_size=node_hidden_size, num_prop_rounds=num_prop_rounds)
    opt = Adam(model.parameters(), lr=lr)

    num_batches = max(1, len(ds) // batch_size)
    printer = CyclePrinting(num_epochs=nepochs, num_batches=num_batches)

    model.train()
    batch_count = 0; batch_loss = 0.0
    opt.zero_grad()

    for epoch in range(nepochs):
        for _, data in enumerate(dl):
            log_prob = model(actions=data)
            loss = -log_prob / batch_size
            loss.backward()

            # anti-NaN
            if not torch.isfinite(loss):
                print("[warn] loss NaN/Inf; pulando batch")
                opt.zero_grad(); continue

            clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); opt.zero_grad()

            batch_loss += loss.item()
            batch_count += 1
            if batch_count % batch_size == 0:
                printer.update(epoch + 1, {"averaged_loss": batch_loss})
                batch_loss = 0.0

    os.makedirs(outdir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"v_max": max_size, "node_hidden_size": node_hidden_size, "num_prop_rounds": num_prop_rounds},
    }
    torch.save(ckpt, os.path.join(outdir, "model.pt"))
    print(f"saved: {os.path.join(outdir, 'model.pt')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets-dir', type=str, default='./runs_star/datasets')
    ap.add_argument('--orderings', type=str, default='degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral')
    ap.add_argument('--logroot', type=str, default='./runs_star')
    ap.add_argument('--nepochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--node-hidden-size', type=int, default=16)
    ap.add_argument('--num-prop-rounds', type=int, default=2)
    ap.add_argument('--max-size', type=int, default=20)
    args = ap.parse_args()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    outroot = os.path.join(args.logroot, f"train_orders_star_{ts}")
    mkdir_p(outroot)
    print(f"Usando: {outroot}")

    orderings = [o.strip() for o in args.orderings.split(',') if o.strip()]
    for o in orderings:
        fname = os.path.join(args.datasets_dir, f'star_{o}.p')
        if not os.path.exists(fname):
            print(f"[warn] dataset não encontrado para '{o}': {fname} — pulando.")
            continue
        outdir = os.path.join(outroot, o)
        print(f"--> training {o} from {fname}")
        train_one(
            fname, outdir,
            max_size=args.max_size,
            node_hidden_size=args.node_hidden_size,
            num_prop_rounds=args.num_prop_rounds,
            lr=args.lr,
            nepochs=args.nepochs,
            batch_size=args.batch_size
        )

if __name__ == '__main__':
    main()
