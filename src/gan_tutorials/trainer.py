## filename: trainer.py

import os
import sys
from tqdm import tqdm


def train(model, dataloader):
    model.global_epoch += 1
    model.train()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Train", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            batch_size = batch["image"].shape[0]
            total += batch_size

            outputs = model.train_step(batch)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def evaluate(model, dataloader):
    model.eval()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Evaluate", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            batch_size = batch["image"].shape[0]
            total += batch_size

            outputs = model.eval_step(batch)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def fit(model, train_loader, num_epochs, total_epochs=None, valid_loader=None):
    history = {"train": {}, "valid": {}}
    for epoch in range(1, num_epochs + 1):
        train_results = train(model, train_loader)
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in train_results.items()])

        if hasattr(model, 'global_epoch') and total_epochs is not None:
            epoch_info = f"[{model.global_epoch:3d}/{total_epochs}]"
        else:
            epoch_info = f"[{epoch:3d}/{num_epochs}]"

        for name, value in train_results.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)
            valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in valid_results.items()])

            for name, value in valid_results.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)
            print(f"{epoch_info} {train_info} | (val) {valid_info}")
        else:
            print(f"{epoch_info} {train_info}")

    return history
