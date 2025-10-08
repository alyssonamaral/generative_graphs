# train_orders_binary.py
import argparse
import os
import time
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from model import DGMG
from binary_trees import BinaryTreeDataset
from cycles import CyclePrinting  # reuso do seu printer
from utils import mkdir_p

def train_one(fname, outdir, max_size, node_hidden_size, num_prop_rounds, lr, nepochs, batch_size):
    ds = BinaryTreeDataset(fname=fname)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=ds.collate_single)

    model = DGMG(v_max=max_size, node_hidden_size=node_hidden_size, num_prop_rounds=num_prop_rounds)
    opt = Adam(model.parameters(), lr=lr)

    batches_per_epoch = len(ds) // batch_size if len(ds) % batch_size == 0 else (len(ds)//batch_size + 1)
    printer = CyclePrinting(num_epochs=nepochs, num_batches=batches_per_epoch)

    model.train()
    for epoch in range(nepochs):
        opt.zero_grad()
        batch_count = 0; batch_loss = 0; batch_prob = 0
        for i, data in enumerate(dl):
            log_prob = model(actions=data)
            loss = -log_prob / batch_size
            loss.backward()
            batch_loss += loss.item()
            batch_prob += log_prob.detach().exp().item() / batch_size
            batch_count += 1
            if batch_count % batch_size == 0:
                clip_grad_norm_(model.parameters(), 0.25)
                opt.step()
                opt.zero_grad()
                printer.update(epoch+1, {"averaged_loss": batch_loss, "averaged_prob": batch_prob})
                batch_loss = 0; batch_prob = 0
    os.makedirs(outdir, exist_ok=True)
    torch.save(model, os.path.join(outdir, "model.pth"))
    print(f"saved: {os.path.join(outdir, 'model.pth')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets-dir', type=str, default='./runs_bin/datasets')
    ap.add_argument('--orderings', type=str, default='degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral')
    ap.add_argument('--logroot', type=str, default='./runs_bin')
    ap.add_argument('--nepochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--node-hidden-size', type=int, default=16)
    ap.add_argument('--num-prop-rounds', type=int, default=2)
    ap.add_argument('--max-size', type=int, default=20)
    args = ap.parse_args()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    outroot = os.path.join(args.logroot, f"train_orders_binary_{ts}")
    mkdir_p(outroot)
    print(f"Usando: {outroot}")

    orderings = [o.strip() for o in args.orderings.split(',') if o.strip()]
    for o in orderings:
        fname = os.path.join(args.datasets_dir, f'bin_trees_{o}.p')
        outdir = os.path.join(outroot, o)
        print(f"--> training {o} from {fname}")
        train_one(fname, outdir, args.max_size, args.node_hidden_size, args.num_prop_rounds, args.lr, args.nepochs, args.batch_size)

if __name__ == '__main__':
    main()
