"""Train EEG-Conformer for the subject-A01 value reported in Table 2.

The model and protocol come from the CTNet authors' fine-tuned Conformer
notebook.  This run uses A01T -> A01E, seed 1715, 1,000 epochs, S&R
augmentation, and the notebook's validation-loss checkpoint selection.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from CTNet import (
    FeedForwardBlock,
    MultiHeadAttention,
    parse_training_args,
    train_model,
)


class ConformerResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.fn(x)
        x += residual
        return x


class ConformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 40, heads: int = 10) -> None:
        super().__init__(
            ConformerResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, heads, 0.5),
                    nn.Dropout(0.5),
                )
            ),
            ConformerResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=4, dropout=0.5),
                    nn.Dropout(0.5),
                )
            ),
        )


class Conformer(nn.Module):
    """The exact A01 architecture in Conformer_fine_tune_2a_77.66_2b_85.87."""

    def __init__(self) -> None:
        super().__init__()
        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (22, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1)),
            # Keep this local rather than adding a third helper script.
            _FlattenPatches(),
        )
        self.transformer = nn.Sequential(
            *[ConformerEncoderBlock(emb_size=40, heads=10) for _ in range(6)]
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2440, 4))

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(self.shallow_net(x))
        x = self.transformer(x)
        return self.classifier(x.flatten(start_dim=1))


class _FlattenPatches(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(start_dim=2).transpose(1, 2)


def build_conformer() -> nn.Module:
    return Conformer()


def main() -> None:
    args = parse_training_args(
        description="Reproduce the Conformer A01 subject-specific Table 2 run.",
        default_seed=1715,
    )
    train_model(
        args,
        model_name="Conformer",
        paper_accuracy=85.07,
        build_model=build_conformer,
    )


if __name__ == "__main__":
    main()
