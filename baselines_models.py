# -*- coding: utf-8 -*-
"""
Arquitecturas de baseline para comparacion controlada.
  - EEGNet  (Lawhern et al. 2018) adaptado a 8 canales, clasificacion binaria
  - CNNOnly  (backbone CNN del modelo propuesto, sin Transformer)

Mismos hiperparametros de entrenamiento que el modelo propuesto:
  epochs=60, lr=5e-4, warmup=4, patience=8, batch=64, seed=42
"""

import copy
import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# EEGNet
# Ref: Lawhern et al. (2018) "EEGNet: a compact convolutional neural network
#      for EEG-based brain-computer interfaces", J. Neural Eng.
# Adaptado para: n_ch=8, T=961 (160Hz * 6s + 1), n_cls=2
# ──────────────────────────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    """
    EEGNet compacto para clasificacion de imageria motora binaria.
    Arquitectura: DepthwiseConv2D + SeparableConv2D + Clasificador lineal.
    """
    def __init__(self, n_ch=8, n_cls=2, sfreq=160, T=961,
                 F1=8, D=2, F2=16, p_drop=0.5, kern_len=None):
        super().__init__()
        if kern_len is None:
            kern_len = sfreq // 2  # 80 para 160 Hz

        # Block 1: Temporal conv + Depthwise spatial conv
        self.block1 = nn.Sequential(
            # Temporal filtering
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Spatial (depthwise) filtering
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=p_drop),
        )

        # Block 2: Separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=p_drop),
        )

        # Calcular dimension aplanada dinamicamente
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, T)
            out = self.block2(self.block1(dummy))
            flat_dim = out.numel()

        self.classifier = nn.Linear(flat_dim, n_cls)

    def forward(self, x):
        # x: (B, C, T)  ->  agregar dim de "imagen": (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────────────
# CNN-Only (backbone del modelo propuesto, sin Transformer)
# Exactamente el mismo stem + bloques depthwise, reemplazando CLS+Attention
# por Global Average Pooling -> Linear head
# ──────────────────────────────────────────────────────────────────────────────
def _make_gn(num_channels, num_groups=8):
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class _DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, p_drop=0.2):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm = _make_gn(out_ch)
        self.act = nn.ELU()
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.pw(self.dw(x)))))


class CNNOnly(nn.Module):
    """
    Variante CNN-only del modelo propuesto.
    Identico al backbone CNN (stem + 2 bloques depthwise separables),
    pero sustituye el Transformer + CLS token por Global Average Pooling.
    """
    def __init__(self, n_ch=8, n_cls=2, p_drop=0.2,
                 k_list=(31, 15), s_list=(2, 2), p_list=(15, 7)):
        super().__init__()
        stem = [
            nn.Conv1d(n_ch, 32, kernel_size=129, stride=2, padding=64, bias=False),
            _make_gn(32),
            nn.ELU(),
            nn.Dropout(p=p_drop),
        ]
        blocks = []
        in_c, out_cs = 32, [64, 128]
        for i in range(2):
            blocks.append(_DepthwiseSepConv(
                in_c, out_cs[i],
                k=k_list[i], s=s_list[i], p=p_list[i],
                p_drop=p_drop,
            ))
            in_c = out_cs[i]

        self.cnn = nn.Sequential(*stem, *blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, n_cls),
        )

    def forward(self, x):
        z = self.cnn(x)          # (B, C', T')
        z = z.mean(dim=-1)       # Global Average Pooling -> (B, C')
        return self.head(z)
