"""Train CTNet for the subject-A01 value reported in Table 2.

This is a deliberately narrow reproduction of the authors' saved 2024
notebook run: BCI Competition IV-2a, A01T -> A01E, seed 966, 1,000 epochs,
S&R augmentation, and validation-loss checkpoint selection.  The default
``legacy`` validation slice preserves the notebook's overlap between the
training and validation samples.  That behavior is methodologically flawed,
but retaining and labelling it is necessary when reproducing the table value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results" / "reproduced_model_trainings"


def default_cache() -> Path:
    """Find the local A01 cache without copying registered competition data."""
    candidates = (
        REPO_ROOT / "Frederik" / "data" / "processed" / "A01.npz",
        REPO_ROOT / "data" / "processed" / "A01.npz",
        REPO_ROOT.parent / "data" / "processed" / "A01.npz",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


class PatchEmbeddingCNN(nn.Module):
    def __init__(
        self,
        f1: int = 8,
        kernel_size: int = 64,
        depth_multiplier: int = 2,
        pooling_size1: int = 8,
        pooling_size2: int = 8,
        dropout_rate: float = 0.5,
        number_channel: int = 22,
    ) -> None:
        super().__init__()
        f2 = depth_multiplier * f1
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f2,
                (number_channel, 1),
                groups=f1,
                padding="valid",
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        self.projection = Rearrange("b e h w -> b (h w) e")

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.cnn_module(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        if mask is not None:
            energy.masked_fill_(~mask, torch.finfo(torch.float32).min)
        attention = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        attention = self.attention_dropout(attention)
        output = torch.einsum("bhal,bhlv->bhav", attention, values)
        return self.projection(rearrange(output, "b h n d -> b n (h d)"))


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, dropout: float = 0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )


class CTNetResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module, emb_size: int, dropout: float) -> None:
        super().__init__()
        self.fn = fn
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_norm(self.dropout(self.fn(x)) + x)


class CTNetEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 16, heads: int = 2) -> None:
        super().__init__(
            CTNetResidualAdd(MultiHeadAttention(emb_size, heads, 0.5), emb_size, 0.5),
            CTNetResidualAdd(FeedForwardBlock(emb_size), emb_size, 0.5),
        )


class CTNet(nn.Module):
    """The exact A01 architecture configured in CTNet_2a_82.91.ipynb."""

    def __init__(self) -> None:
        super().__init__()
        self.emb_size = 16
        self.cnn = PatchEmbeddingCNN()
        self.position_dropout = nn.Dropout(0.1)
        self.position_encoding = nn.Parameter(torch.randn(1, 100, self.emb_size))
        self.transformer = nn.Sequential(
            *[CTNetEncoderBlock(self.emb_size, heads=2) for _ in range(6)]
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(240, 4))

    def forward(self, x: Tensor) -> Tensor:
        cnn = self.cnn(x) * math.sqrt(self.emb_size)
        cnn = self.position_dropout(cnn + self.position_encoding[:, : cnn.shape[1]])
        features = cnn + self.transformer(cnn)
        return self.classifier(features.flatten(start_dim=1))


def build_ctnet() -> nn.Module:
    return CTNet()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_driver() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip().splitlines()[0]


def segmentation_reconstruction(
    data: np.ndarray,
    labels_one_based: np.ndarray,
    *,
    n_aug: int,
    n_segments: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Apply the notebook's segmentation-and-reconstruction augmentation."""
    augmented_data: list[np.ndarray] = []
    augmented_labels: list[np.ndarray] = []
    records_per_class = n_aug * int(batch_size / 4)
    segment_width = 1000 // n_segments

    for class_id in range(1, 5):
        class_data = data[labels_one_based == class_id]
        class_labels = labels_one_based[labels_one_based == class_id]
        generated = np.zeros(
            (records_per_class, 1, 22, 1000), dtype=np.float64
        )
        for record in range(records_per_class):
            for segment in range(n_segments):
                # The notebook draws a fresh vector for every segment and then
                # uses only the element at that segment's index.  Keeping the
                # otherwise redundant draws is required to preserve its seeded
                # NumPy random stream exactly.
                source_trials = np.random.randint(0, class_data.shape[0], n_segments)
                source_trial = source_trials[segment]
                left = segment * segment_width
                right = (segment + 1) * segment_width
                generated[record, :, :, left:right] = class_data[
                    source_trial, :, :, left:right
                ]
        augmented_data.append(generated)
        augmented_labels.append(class_labels[:records_per_class])

    data_array = np.concatenate(augmented_data)
    label_array = np.concatenate(augmented_labels)
    order = np.random.permutation(len(data_array))
    return (
        torch.from_numpy(data_array[order]).float().to(device),
        torch.from_numpy(label_array[order] - 1).long().to(device),
    )


def train_model(
    args: argparse.Namespace,
    *,
    model_name: str,
    paper_accuracy: float,
    build_model: Callable[[], nn.Module],
) -> dict[str, object]:
    if not args.cache.exists():
        raise FileNotFoundError(
            f"A01 cache not found at {args.cache}. Pass --cache path/to/A01.npz."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this Python environment."
        )

    cached = np.load(args.cache)
    train_data = cached["train"]
    train_labels = cached["train_labels"].reshape(-1).astype(np.int64)
    test_data = cached["test"]
    test_labels = cached["test_labels"].reshape(-1).astype(np.int64) - 1
    if train_data.shape != (288, 22, 1000) or test_data.shape != (288, 22, 1000):
        raise RuntimeError(
            "Expected A01 train/test arrays with shape (288, 22, 1000); "
            f"got {train_data.shape} and {test_data.shape}."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    model = build_model().to(device)

    # Match get_source_data(): shuffle once, then use global A01T statistics.
    order = np.random.permutation(len(train_data))
    all_data = train_data[order, None]
    all_labels = train_labels[order]
    train_mean = float(all_data.mean())
    train_std = float(all_data.std())
    all_data = (all_data - train_mean) / train_std
    standardized_test = (test_data[:, None] - train_mean) / train_std

    dataset = TensorDataset(
        torch.from_numpy(all_data), torch.from_numpy(all_labels - 1)
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name.lower()}_a01"
    checkpoint = output_dir / f"{stem}_state.pt"
    history_path = output_dir / f"{stem}_history.csv"
    result_path = output_dir / f"{stem}_result.json"

    best_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    print(
        f"model={model_name} subject=A01 device={device} seed={args.seed} "
        f"epochs={args.epochs} validation={args.validation}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        validation_data: list[Tensor] = []
        validation_labels: list[Tensor] = []
        last_loss: Tensor | None = None
        last_logits: Tensor | None = None
        last_training_labels: Tensor | None = None

        for batch_data, batch_labels in data_loader:
            validation_count = int(args.validation_ratio * len(batch_data))
            raw_train_data = batch_data[:-validation_count]
            raw_train_labels = batch_labels[:-validation_count]
            if args.validation == "legacy":
                # Published notebook: 51 validation samples and 30 overlap the
                # 51 raw training samples in every full batch of 72.
                validation_data.append(batch_data[validation_count:])
                validation_labels.append(batch_labels[validation_count:])
            else:
                validation_data.append(batch_data[-validation_count:])
                validation_labels.append(batch_labels[-validation_count:])

            train_tensor = raw_train_data.float().to(device)
            label_tensor = raw_train_labels.long().to(device)
            aug_data, aug_labels = segmentation_reconstruction(
                all_data,
                all_labels,
                n_aug=args.n_aug,
                n_segments=args.n_segments,
                batch_size=args.batch_size,
                device=device,
            )
            train_tensor = torch.cat((train_tensor, aug_data))
            label_tensor = torch.cat((label_tensor, aug_labels))
            logits = model(train_tensor)
            loss = criterion(logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss
            last_logits = logits
            last_training_labels = label_tensor

        assert last_loss is not None
        assert last_logits is not None
        assert last_training_labels is not None

        model.eval()
        val_data = torch.cat(validation_data).float()
        val_labels = torch.cat(validation_labels).long().to(device)
        val_outputs: list[Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(val_data), args.batch_size):
                val_outputs.append(
                    model(val_data[start : start + args.batch_size].to(device))
                )
        val_logits = torch.cat(val_outputs)
        val_loss = criterion(val_logits, val_labels)
        val_accuracy = float((val_logits.argmax(1) == val_labels).float().mean())
        train_accuracy = float(
            (last_logits.argmax(1) == last_training_labels).float().mean()
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss_last_batch": float(last_loss.detach().cpu()),
            "train_accuracy_last_batch": train_accuracy,
            "validation_loss": float(val_loss.detach().cpu()),
            "validation_accuracy": val_accuracy,
        }
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_loss = float(record["validation_loss"])
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
            print(
                f"epoch={epoch:04d} train_acc={train_accuracy:.4f} "
                f"val_acc={val_accuracy:.4f} val_loss={best_loss:.7f}",
                flush=True,
            )

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    test_outputs: list[Tensor] = []
    with torch.inference_mode():
        test_tensor = torch.from_numpy(standardized_test).float()
        for start in range(0, len(test_tensor), args.batch_size):
            test_outputs.append(
                model(test_tensor[start : start + args.batch_size].to(device)).cpu()
            )
    predictions = torch.cat(test_outputs).argmax(1).numpy()
    correct = int((predictions == test_labels).sum())
    accuracy = correct / len(test_labels)
    elapsed = time.perf_counter() - started

    result: dict[str, object] = {
        "experiment": "training from scratch",
        "model": model_name,
        "dataset": "BCI Competition IV-2a",
        "subject": "A01",
        "training_recording": "A01T",
        "evaluation_recording": "A01E",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch_zero_based": best_epoch,
        "best_validation_loss": best_loss,
        "validation_implementation": args.validation,
        "validation_note": (
            "2024 notebook overlap and per-epoch reshuffle preserved"
            if args.validation == "legacy"
            else "overlap removed, but per-epoch reshuffle preserved"
        ),
        "batch_size_raw": args.batch_size,
        "augmentation_factor": args.n_aug,
        "augmentation_segments": args.n_segments,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "correct": correct,
        "n_test_trials": int(len(test_labels)),
        "accuracy": accuracy,
        "accuracy_percent": 100 * accuracy,
        "cohen_kappa": float(cohen_kappa_score(test_labels, predictions)),
        "paper_table_2_accuracy_percent": paper_accuracy,
        "absolute_difference_percentage_points": abs(100 * accuracy - paper_accuracy),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            test_labels, predictions, labels=[0, 1, 2, 3]
        ).tolist(),
        "normalization": {"a01t_mean": train_mean, "a01t_std": train_std},
        "cache": {"path": str(args.cache.resolve()), "sha256": sha256(args.cache)},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "nvidia_driver": gpu_driver(),
        },
        "files": {
            "checkpoint": str(checkpoint),
            "history": str(history_path),
            "result": str(result_path),
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_training_args(
    *, description: str, default_seed: int, default_device: str = "cuda"
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--n-aug", type=int, default=3)
    parser.add_argument("--n-segments", type=int, default=8)
    parser.add_argument("--validation-ratio", type=float, default=0.3)
    parser.add_argument(
        "--validation", choices=("legacy", "corrected"), default="legacy"
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_training_args(
        description="Reproduce the CTNet A01 subject-specific Table 2 run.",
        default_seed=966,
    )
    train_model(
        args,
        model_name="CTNet",
        paper_accuracy=90.97,
        build_model=build_ctnet,
    )


if __name__ == "__main__":
    main()
