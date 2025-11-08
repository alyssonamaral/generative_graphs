# dgmg/train_orders_ba.py
import argparse
import os
import pickle
import time
import datetime

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from model import DGMG

class SeqDataset(Dataset):
    def __init__(self, pfile):
        with open(pfile, "rb") as f:
            self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    @staticmethod
    def collate_single(batch):
        assert len(batch) == 1
        return batch[0]

def train_one(pfile, outdir, max_size, node_hidden=16, num_prop_rounds=2,
              nepochs=25, batch_size=1, lr=1e-4, clip_bound=0.25, device="cpu"):
    ds = SeqDataset(pfile)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                    collate_fn=SeqDataset.collate_single)

    model = DGMG(v_max=max_size, node_hidden_size=node_hidden,
                 num_prop_rounds=num_prop_rounds)
    model.to(device)
    model.train()

    opt = Adam(model.parameters(), lr=lr)
    t0 = time.time()
    num_batches = len(ds) // batch_size if len(ds) >= batch_size else 1

    running_loss = 0.0
    running_prob = 0.0
    count = 0
    opt.zero_grad()

    for epoch in range(1, nepochs+1):
        for i, actions in enumerate(dl, start=1):
            logp = model(actions=actions)
            prob = logp.detach().exp()
            loss = -logp / batch_size
            (loss).backward()

            running_loss += loss.item()
            running_prob += (prob / batch_size).item()
            count += 1

            if count % batch_size == 0:
                clip_grad_norm_(model.parameters(), clip_bound)
                opt.step()
                opt.zero_grad()

        avgL = running_loss
        avgP = running_prob
        print(f"epoch {epoch}/{nepochs}, batches={num_batches}, averaged_loss: {avgL:.6f}, averaged_prob: {avgP:.6f}")
        running_loss = 0.0
        running_prob = 0.0

    os.makedirs(outdir, exist_ok=True)
    torch.save(model, os.path.join(outdir, "model.pth"))
    print(f"Salvo modelo em {outdir}/model.pth | tempo={(time.time()-t0):.1f}s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", type=str, default="./runs_ba/datasets")
    ap.add_argument("--orderings", type=str, default="degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral")
    ap.add_argument("--logroot", type=str, default="./runs_ba")
    ap.add_argument("--nepochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)  # mais estável para BA
    ap.add_argument("--node-hidden-size", type=int, default=16)
    ap.add_argument("--num-prop-rounds", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=60)
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outroot = os.path.join(args.logroot, f"train_orders_ba_{ts}")
    os.makedirs(outroot, exist_ok=True)

    print(f"Usando: {outroot}")
    orderings = [o.strip() for o in args.orderings.split(",") if o.strip()]

    for o in orderings:
        src = os.path.join(args.datasets_dir, f"ba_{o}.p")
        out = os.path.join(outroot, o)
        print(f"\n=== Treinando ordenação: {o} ===")
        train_one(src, out, max_size=args.max_size,
                  node_hidden=args.node_hidden_size,
                  num_prop_rounds=args.num_prop_rounds,
                  nepochs=args.nepochs, batch_size=args.batch_size,
                  lr=args.lr)

if __name__ == "__main__":
    main()
