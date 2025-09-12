import argparse
import numpy as np
import torch
from .models.vgae_graphon import VGAEGraphon
from .utils import ring_positions, is_cycle_adj, save_report

def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    model = VGAEGraphon(in_dim=a['in_dim'], hidden=a['hidden'], out_dim=a['hidden'],
    z_g_dim=a['z_g'], pe_K=a['pe_K'], dec_hidden=a['dec_hidden']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, a

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, a = load_model(args.ckpt, device)

    num_samples = args.num_samples
    v_min, v_max = args.min_size, args.max_size

    sizes = np.random.randint(v_min, v_max+1, size=num_samples)
    total_n = sizes.sum()

    num_valid_size = 0
    num_cycles = 0


    for N in sizes:
        s = ring_positions(N).to(device)
        z = torch.randn(1, a['z_g'], device=device)
        logits = model.decode_logits(z, s)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        P = np.triu(probs, k=1)
        A = (np.random.rand(N, N) < P).astype(np.float32)
        A = A + A.T
        num_valid_size += 1 # por construção N está no intervalo
        num_cycles += int(is_cycle_adj(torch.from_numpy(np.array(A))))

    stats = {
    'num_samples': int(num_samples),
    'v_min': int(v_min),
    'v_max': int(v_max),
    'average_size': float(sizes.mean()),
    'valid_size_ratio': float(num_valid_size/num_samples),
    'cycle_ratio': float(num_cycles/num_samples),
    'valid_ratio': float(num_cycles/num_samples),
    }
    if args.report:
        save_report(args.report, stats)
        print(f"Saved report to {args.report}")
    else:
        print(stats)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--min-size', type=int, default=5)
    p.add_argument('--max-size', type=int, default=20)
    p.add_argument('--report', type=str, default='')
    args = p.parse_args()
    evaluate(args)