# configure.py

def dataset_based_configure(opts):
    dataset_map = {
        "cycles": cycles_configure,
        "trees":  trees_configure,
    }
    if opts["dataset"] not in dataset_map:
        raise ValueError(f"Unsupported dataset: {opts['dataset']}")

    # merge (CLI > defaults)
    merged = {**dataset_map[opts["dataset"]], **opts}

    # default de path se não veio da CLI
    if not merged.get("path_to_dataset"):
        merged["path_to_dataset"] = (
            "cycles_small.p" if merged["dataset"] == "cycles" else "trees_small.p"
        )
    return merged


synthetic_dataset_configure = {
    "node_hidden_size": 16,
    "num_propagation_rounds": 2,
    "optimizer": "Adam",
    "nepochs": 25,
    "ds_size": 4000,
    "num_generated_samples": 10000,
    "generation_mode": "general",
}

cycles_configure = {
    **synthetic_dataset_configure,
    **{
        "min_size": 10,
        "max_size": 20,
        "lr": 5e-4,
        "generation_mode": "general",
    },
}

trees_configure = {
    **synthetic_dataset_configure,
    **{
        "min_size": 5,
        "max_size": 20,
        "lr": 1e-3,
        "generation_mode": "tree",
    },
}
