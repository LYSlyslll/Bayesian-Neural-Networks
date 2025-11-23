"""Evaluate an MC Dropout MNIST model and export predictions/uncertainty."""
from __future__ import print_function, division

import argparse
import json
from pathlib import Path

import torch
import torch.utils.data
from torchvision import datasets, transforms

from src.MC_dropout.model import MC_drop_net
from src.utils import mkdir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MC Dropout network on MNIST and optionally save predictions"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="MCdrop_models/theta_best.dat",
        help="Path to the saved checkpoint produced by train_MCDropout_MNIST.py",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for evaluation (default: 512)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=20,
        help="Number of MC Dropout forward passes per batch (default: 20)",
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        help="Compute the loss on logits averaged across samples instead of averaging softmax outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="MCdrop_predictions",
        help="Where to save metrics and prediction arrays",
    )
    parser.add_argument(
        "--save_sample_probs",
        action="store_true",
        help="If set, saves per-sample class probabilities for each MC forward pass",
    )
    return parser.parse_args()


def build_dataloader(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    test_set = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    use_cuda = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=3,
    )
    return loader, use_cuda


def load_model(weights_path, use_cuda):
    net = MC_drop_net(
        lr=1e-3,
        channels_in=1,
        side_in=28,
        cuda=use_cuda,
        classes=10,
        batch_size=128,
        weight_decay=1,
        n_hid=1200,
    )

    map_location = None if use_cuda else "cpu"
    net.load(weights_path, map_location=map_location)
    net.set_mode_train(False)
    return net


def evaluate(net, loader, nsamples, logits, save_sample_probs):
    total_loss = 0.0
    total_err = 0.0
    total_examples = 0
    mean_probs_list = []
    pred_labels = []
    sample_probs = [] if save_sample_probs else None

    with torch.no_grad():
        for x, y in loader:
            if nsamples > 1:
                loss, batch_err, probs = net.sample_eval(x, y, Nsamples=nsamples, logits=logits)
                if save_sample_probs:
                    sample_probs.append(net.all_sample_eval(x, y, Nsamples=nsamples).cpu())
            else:
                loss, batch_err, probs = net.eval(x, y)

            total_examples += len(x)
            total_loss += loss.item()
            total_err += batch_err.item()

            mean_probs_list.append(probs)
            pred_labels.append(probs.max(dim=1)[1])

    mean_probs = torch.cat(mean_probs_list, dim=0)
    preds = torch.cat(pred_labels, dim=0)

    metrics = {
        "nll": float(total_loss / total_examples),
        "error_rate": float(total_err / total_examples),
        "accuracy": float(1.0 - total_err / total_examples),
        "examples": int(total_examples),
        "nsamples": int(nsamples),
    }

    results = {
        "metrics": metrics,
        "pred_labels": preds.cpu().numpy(),
        "mean_probs": mean_probs.cpu().numpy(),
    }

    if save_sample_probs and sample_probs is not None:
        results["sample_probs"] = torch.cat(sample_probs, dim=1).cpu().numpy()

    return results


def save_outputs(results, output_dir):
    mkdir(output_dir)
    out_path = Path(output_dir)

    with open(out_path / "metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)

    npz_kwargs = {"pred_labels": results["pred_labels"], "mean_probs": results["mean_probs"]}
    if "sample_probs" in results:
        npz_kwargs["sample_probs"] = results["sample_probs"]
    torch.save(npz_kwargs, out_path / "predictions.pt")

    print("Saved metrics to", out_path / "metrics.json")
    print("Saved predictions to", out_path / "predictions.pt")


if __name__ == "__main__":
    import numpy as np

    args = parse_args()
    loader, use_cuda = build_dataloader(batch_size=args.batch_size)
    net = load_model(args.weights, use_cuda)
    results = evaluate(
        net,
        loader,
        nsamples=args.nsamples,
        logits=args.logits,
        save_sample_probs=args.save_sample_probs,
    )
    save_outputs(results, args.output_dir)

    print("Metrics:")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")
