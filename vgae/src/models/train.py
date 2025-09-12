import argparse
import os
import torch
from torch.utils.data import DataLoader
from .utils import set_seed, ensure_dir
from .datasets.cycles import CyclesDataset
from .models.vgae_graphon import VGAEGraphon

def train(args):
    ds = CyclesDataset(num_graphs=args.num_graphs, min_size=args.min_size, max_size=args.max_size,
    in_dim=args.in_dim, seed=args.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=CyclesDataset.collate_single)

    model = VGAEGraphon(in_dim=args.in_dim, hidden=args.hidden, out_dim=args.hidden,
    z_g_dim=args.z_g, pe_K=args.pe_K, dec_hidden=args.dec_hidden).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ensure_dir(args.logdir)
    best_loss = float('inf')


    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for item in dl:
            A = item['A'].to(device)
            X = item['X'].to(device)
            s = item['s'].to(device)
            opt.zero_grad()
            loss, recon, kl, _ = model(X, A, s, beta_kl=args.beta_kl, lambda_deg=args.lambda_deg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()
        avg = running / len(dl)
        print(f"epoch {epoch}/{args.epochs} | loss {avg:.4f}")
        # checkpoint simples
        if avg < best_loss:
            best_loss = avg
            ckpt = {
            'state_dict': model.state_dict(),
            'args': vars(args),
            }
            torch.save(ckpt, os.path.join(args.logdir, 'model.pt'))
            print(f" saved best model @ {args.logdir}/model.pt")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['cycles'], default='cycles')
    p.add_argument('--min-size', type=int, default=5)
    p.add_argument('--max-size', type=int, default=20)
    p.add_argument('--num-graphs', type=int, default=4000)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--z-g', type=int, default=32)
    p.add_argument('--pe-K', type=int, default=2)
    p.add_argument('--dec-hidden', type=int, default=64)
    p.add_argument('--in-dim', type=int, default=1)
    p.add_argument('--beta-kl', type=float, default=1.0)
    p.add_argument('--lambda-deg', type=float, default=0.0)
    p.add_argument('--logdir', type=str, default='./runs/cycles')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    train(args)