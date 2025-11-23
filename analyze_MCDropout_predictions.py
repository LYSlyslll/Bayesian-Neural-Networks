"""可视化 evaluate_MCDropout_MNIST.py 生成的 predictions.pt。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="可视化 evaluate_MCDropout_MNIST.py 导出的 predictions.pt 与 metrics.json",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="MCdrop_predictions/predictions.pt",
        help="evaluate_MCDropout_MNIST.py 导出的 predictions.pt 路径",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="metrics.json 路径（默认与 predictions 同目录）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data",
        help="MNIST 数据根目录，用于读取测试集标签/图像以对齐预测",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="MCdrop_predictions/analysis",
        help="可视化结果输出目录",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=15,
        help="可靠度曲线与 ECE 计算使用的置信度分桶数量",
    )
    parser.add_argument(
        "--top_uncertain",
        type=int,
        default=16,
        help="保存预测熵最高的前 N 张测试集图像 (默认 16)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="保存图像时的 DPI",
    )
    return parser.parse_args()


def load_predictions(pred_path: Path) -> Dict[str, np.ndarray]:
    # PyTorch 2.6 起默认 weights_only=True，明确关闭以兼容现有 pt 文件
    raw = torch.load(pred_path, map_location="cpu", weights_only=False)
    return {k: v if isinstance(v, np.ndarray) else np.asarray(v) for k, v in raw.items()}


def load_mnist_labels(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (labels, images)；顺序与 evaluate 脚本一致（shuffle=False）。"""

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    labels = np.array([label for _, label in test_set], dtype=np.int64)
    images = np.stack([np.array(img.squeeze()) for img, _ in test_set], axis=0)
    return labels, images


def compute_confusion(preds: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    return conf


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, num_bins: int):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = predictions == labels

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1

    bin_acc = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    bin_count = np.zeros(num_bins)

    for b in range(num_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_acc[b] = correctness[mask].mean()
        bin_conf[b] = confidences[mask].mean()
        bin_count[b] = mask.sum()

    ece = np.sum((bin_count / len(probs)) * np.abs(bin_acc - bin_conf))
    return bins, bin_acc, bin_conf, bin_count, ece


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def mutual_information(mean_probs: np.ndarray, sample_probs: np.ndarray) -> np.ndarray:
    # sample_probs: [S, N, C]
    mean_entropy = predictive_entropy(mean_probs)
    sample_entropy = -np.sum(sample_probs * np.log(sample_probs + 1e-12), axis=2)
    expected_entropy = sample_entropy.mean(axis=0)
    return mean_entropy - expected_entropy


def plot_confusion(conf: np.ndarray, output: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_reliability(bins, bin_acc, bin_conf, output: Path, dpi: int, ece: float):
    fig, ax = plt.subplots(figsize=(6, 5))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(bin_centers, bin_acc, width=1.0 / len(bin_centers), alpha=0.6, label="准确率")
    ax.plot([0, 1], [0, 1], "k--", label="完美校准")
    ax.set_xlabel("置信度")
    ax.set_ylabel("准确率")
    ax.set_title(f"可靠度曲线 / ECE={ece:.4f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_hist(data: np.ndarray, title: str, xlabel: str, output: Path, dpi: int, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, color="#4C72B0", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("频数")
    plt.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def save_top_uncertain_images(
    images: np.ndarray,
    labels: np.ndarray,
    mean_probs: np.ndarray,
    entropy: np.ndarray,
    output: Path,
    top_k: int,
    dpi: int,
):
    if top_k <= 0:
        return

    idx = np.argsort(-entropy)[:top_k]
    cols = int(math.sqrt(top_k))
    cols = max(cols, 1)
    rows = int(math.ceil(top_k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)

    for ax, i in zip(axes.flat, idx):
        ax.imshow(images[i], cmap="gray")
        pred = mean_probs[i].argmax()
        conf = mean_probs[i].max()
        ax.set_title(f"真:{labels[i]} 预测:{pred}\nconf={conf:.2f} H={entropy[i]:.2f}")
        ax.axis("off")

    # 关闭未用子图坐标轴
    for ax in axes.flat[len(idx) :]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    metrics_path = Path(args.metrics) if args.metrics else pred_path.parent / "metrics.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_dict = load_predictions(pred_path)
    labels, images = load_mnist_labels(Path(args.data_root))

    mean_probs = preds_dict["mean_probs"]
    pred_labels = preds_dict["pred_labels"].astype(np.int64)
    assert len(mean_probs) == len(labels), "预测数量与测试集大小不一致"

    # 基础指标
    conf_matrix = compute_confusion(pred_labels, labels)
    acc = (pred_labels == labels).mean()

    bins, bin_acc, bin_conf, bin_count, ece = reliability_diagram(
        mean_probs, labels, num_bins=args.num_bins
    )

    entropy = predictive_entropy(mean_probs)
    confidence = mean_probs.max(axis=1)

    sample_probs = preds_dict.get("sample_probs")
    mi = None
    if sample_probs is not None:
        mi = mutual_information(mean_probs, sample_probs)

    # 保存图像
    plot_confusion(conf_matrix, output_dir / "confusion_matrix.png", dpi=args.dpi)
    plot_reliability(bins, bin_acc, bin_conf, output_dir / "reliability.png", dpi=args.dpi, ece=ece)
    plot_hist(confidence, "预测置信度分布", "最大类别概率", output_dir / "confidence_hist.png", dpi=args.dpi)
    plot_hist(entropy, "预测熵分布", "熵", output_dir / "entropy_hist.png", dpi=args.dpi)
    if mi is not None:
        plot_hist(mi, "互信息分布 (不确定性)", "互信息", output_dir / "mutual_information_hist.png", dpi=args.dpi)
    save_top_uncertain_images(
        images,
        labels,
        mean_probs,
        entropy,
        output_dir / "top_uncertain.png",
        top_k=args.top_uncertain,
        dpi=args.dpi,
    )

    # 保存衍生指标，便于后续分析
    np.savez(
        output_dir / "analysis_stats.npz",
        accuracy=acc,
        ece=ece,
        bins=bins,
        bin_accuracy=bin_acc,
        bin_confidence=bin_conf,
        bin_count=bin_count,
        entropy=entropy,
        confidence=confidence,
        mutual_information=mi,
    )

    print("================ 评估概览 ================")
    if metrics_path.exists():
        import json

        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print("metrics.json:", metrics)
    print(f"预测文件: {pred_path}")
    print(f"准确率(基于 mean_probs): {acc:.4f}")
    print(f"ECE (num_bins={args.num_bins}): {ece:.4f}")
    print(f"熵: mean={entropy.mean():.4f}, std={entropy.std():.4f}")
    if mi is not None:
        print(f"互信息(不确定性): mean={mi.mean():.4f}, std={mi.std():.4f}")
    print("图像/统计已保存到:", output_dir)


if __name__ == "__main__":
    main()
