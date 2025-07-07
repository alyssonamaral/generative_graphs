"""
Learning Deep Generative Models of Graphs
Paper: https://arxiv.org/pdf/1803.03324.pdf

Este script gera automaticamente um dataset pequeno se ele não existir.
"""

import argparse
import datetime
import os
import time

import torch
from model import DGMG
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

# ------------------------------------------------------------------------------
# GERAÇÃO AUTOMÁTICA DO DATASET PEQUENO
# ------------------------------------------------------------------------------
from cycles import generate_dataset

if not os.path.exists("cycles_large.p"):
    print("Gerando dataset cycles_large.p...")
    generate_dataset(
        v_min=10,
        v_max=20,
        n_samples=4000,
        fname="cycles_large.p"
    )
else:
    print("Dataset cycles_large.p já existe.")
# ------------------------------------------------------------------------------

def main(opts):
    t1 = time.time()

    # Setup dataset e data loader
    from cycles import CycleDataset, CycleModelEvaluation, CyclePrinting

    dataset = CycleDataset(fname=opts["path_to_dataset"])
    evaluator = CycleModelEvaluation(
        v_min=opts["min_size"],
        v_max=opts["max_size"],
        dir=opts["log_dir"]
    )
    printer = CyclePrinting(
        num_epochs=opts["nepochs"],
        num_batches=opts["ds_size"] // opts["batch_size"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_single,
    )

    # Inicializa o modelo
    model = DGMG(
        v_max=opts["max_size"],
        node_hidden_size=opts["node_hidden_size"],
        num_prop_rounds=opts["num_propagation_rounds"],
    )

    # Inicializa otimizador
    optimizer = Adam(model.parameters(), lr=opts["lr"])

    t2 = time.time()

    # Treinamento
    model.train()
    for epoch in range(opts["nepochs"]):
        batch_count = 0
        batch_loss = 0
        batch_prob = 0
        optimizer.zero_grad()

        for i, data in enumerate(data_loader):
            log_prob = model(actions=data)
            prob = log_prob.detach().exp()

            loss = -log_prob / opts["batch_size"]
            prob_averaged = prob / opts["batch_size"]

            loss.backward()

            batch_loss += loss.item()
            batch_prob += prob_averaged.item()
            batch_count += 1

            if batch_count % opts["batch_size"] == 0:
                printer.update(
                    epoch + 1,
                    {"averaged_loss": batch_loss, "averaged_prob": batch_prob},
                )

                if opts["clip_grad"]:
                    clip_grad_norm_(model.parameters(), opts["clip_bound"])

                optimizer.step()

                batch_loss = 0
                batch_prob = 0
                optimizer.zero_grad()

    t3 = time.time()

    # Avaliação
    model.eval()
    evaluator.rollout_and_examine(model, opts["num_generated_samples"])
    evaluator.write_summary()

    t4 = time.time()

    print("It took {} to setup.".format(datetime.timedelta(seconds=t2 - t1)))
    print("It took {} to finish training.".format(datetime.timedelta(seconds=t3 - t2)))
    print("It took {} to finish evaluation.".format(datetime.timedelta(seconds=t4 - t3)))
    print("--------------------------------------------------------------------------")
    print("On average, an epoch takes {}.".format(
        datetime.timedelta(seconds=(t3 - t2) / opts["nepochs"])
    ))

    # Salva modelo
    torch.save(model, "./model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGMG")

    # Configurações principais
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Dataset
    parser.add_argument("--dataset", choices=["cycles"], default="cycles", help="Dataset to use")
    parser.add_argument(
        "--path-to-dataset",
        type=str,
        default="cycles_small.p",  # Já aponta para o dataset pequeno
        help="Load the dataset if it exists, generate it otherwise",
    )

    # Log
    parser.add_argument("--log-dir", default="./results_small", help="Folder to save logs and results")

    # Otimização
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip-grad", action="store_true", default=True, help="Enable gradient clipping")
    parser.add_argument("--clip-bound", type=float, default=0.25, help="Gradient norm constraint")

    args = parser.parse_args()

    from utils import setup

    opts = setup(args)

    main(opts)

# ------------------------------------------------------------------------------python main.py \
#
# python main.py \
#  --path-to-dataset cycles_large.p \
#  --batch-size 1 \
#  --log-dir ./results_paper_batch1
#
# ------------------------------------------------------------------------------