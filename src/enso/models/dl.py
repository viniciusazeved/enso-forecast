"""Modelos DL: MLP, LSTM, TCN, Transformer, Mamba (S6 puro PyTorch).

Todos compartilham: input [batch, seq_len=L, n_features=F], output escalar (h
passos a frente, fixado por horizonte da janela). Treinamento com Adam + early
stopping via val_loss.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enso.models.base import ForecastModel


def _train_loop(
    net: nn.Module,
    X: np.ndarray, y: np.ndarray,
    X_val: np.ndarray | None, y_val: np.ndarray | None,
    epochs: int = 300, batch_size: int = 64, lr: float = 1e-3,
    weight_decay: float = 1e-5, patience: int = 25, device: str = "cpu",
    verbose: bool = False,
) -> nn.Module:
    """Loop generico com mini-batches, MSE e early stopping."""
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    if X_val is not None:
        Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
        yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    best, bad = float("inf"), 0
    best_state = None

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            out = net(Xt[idx]).squeeze(-1)
            l = loss_fn(out, yt[idx])
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        if X_val is None:
            continue
        net.eval()
        with torch.no_grad():
            lv = float(loss_fn(net(Xv).squeeze(-1), yv))
        if lv < best - 1e-5:
            best, bad = lv, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop ep={ep} val={best:.4f}")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net


class _MLPNet(nn.Module):
    def __init__(self, lookback: int, in_features: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.flatten = nn.Flatten()
        d = lookback * in_features
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


class MLP(ForecastModel):
    name = "mlp"
    input_kind = "sequence"

    def __init__(self, lookback: int, n_features: int, hidden: int = 128,
                 dropout: float = 0.1, epochs: int = 300, lr: float = 1e-3,
                 batch_size: int = 64, device: str = "cpu"):
        self.lookback = lookback
        self.n_features = n_features
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        self.net_ = _MLPNet(self.lookback, self.n_features, self.hidden, self.dropout)
        self.net_ = _train_loop(
            self.net_, X, y, X_val, y_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size, device=self.device,
        )
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net_(x).squeeze(-1).cpu().numpy()


class _LSTMNet(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTM(ForecastModel):
    name = "lstm"
    input_kind = "sequence"

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, epochs: int = 300, lr: float = 1e-3,
                 batch_size: int = 64, device: str = "cpu"):
        self.n_features, self.hidden, self.layers, self.dropout = n_features, hidden, layers, dropout
        self.epochs, self.lr, self.batch_size, self.device = epochs, lr, batch_size, device

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        self.net_ = _LSTMNet(self.n_features, self.hidden, self.layers, self.dropout)
        self.net_ = _train_loop(
            self.net_, X, y, X_val, y_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size, device=self.device,
        )
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net_(x).squeeze(-1).cpu().numpy()


class _Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__(); self.chomp = chomp
    def forward(self, x):
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp1 = _Chomp1d(pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp2 = _Chomp1d(pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.drop(self.act(self.chomp1(self.conv1(x))))
        out = self.drop(self.act(self.chomp2(self.conv2(out))))
        return self.act(out + self.proj(x))


class _TCNNet(nn.Module):
    def __init__(self, in_features: int, channels: tuple[int, ...] = (64, 64, 64),
                 k: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        c_in = in_features
        for i, c_out in enumerate(channels):
            layers.append(_TCNBlock(c_in, c_out, k, dilation=2 ** i, dropout=dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):
        # x: [B, L, F] -> [B, F, L] for Conv1d
        out = self.tcn(x.transpose(1, 2))
        return self.head(out[:, :, -1])


class TCN(ForecastModel):
    name = "tcn"
    input_kind = "sequence"

    def __init__(self, n_features: int, channels=(64, 64, 64), k: int = 3,
                 dropout: float = 0.1, epochs: int = 300, lr: float = 1e-3,
                 batch_size: int = 64, device: str = "cpu"):
        self.n_features = n_features
        self.channels, self.k, self.dropout = channels, k, dropout
        self.epochs, self.lr, self.batch_size, self.device = epochs, lr, batch_size, device

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        self.net_ = _TCNNet(self.n_features, self.channels, self.k, self.dropout)
        self.net_ = _train_loop(
            self.net_, X, y, X_val, y_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size, device=self.device,
        )
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net_(x).squeeze(-1).cpu().numpy()


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class _TransformerNet(nn.Module):
    def __init__(self, in_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pe = _PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.pe(self.proj(x))
        h = self.enc(h)
        return self.head(h[:, -1, :])


class Transformer(ForecastModel):
    name = "transformer"
    input_kind = "sequence"

    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1,
                 epochs: int = 300, lr: float = 1e-3, batch_size: int = 64,
                 device: str = "cpu"):
        self.n_features, self.d_model, self.nhead = n_features, d_model, nhead
        self.num_layers, self.dim_ff, self.dropout = num_layers, dim_ff, dropout
        self.epochs, self.lr, self.batch_size, self.device = epochs, lr, batch_size, device

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        self.net_ = _TransformerNet(self.n_features, self.d_model, self.nhead,
                                    self.num_layers, self.dim_ff, self.dropout)
        self.net_ = _train_loop(
            self.net_, X, y, X_val, y_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size, device=self.device,
        )
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net_(x).squeeze(-1).cpu().numpy()


class _MambaBlock(nn.Module):
    """Bloco S6 minimalista (Mamba) em PyTorch puro.

    Referencia: Gu & Dao 2023 (S6 / Mamba). Implementacao didatica seguindo o
    repo oficial mamba-minimal (Albert Gu). Sem CUDA kernel - usa scan recursivo
    (mais lento, ok pra series curtas tipo 12 meses).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.transpose(1, 2)  # [B, d_inner, L]
        x_in = self.conv1d(x_in)[:, :, :L]
        x_in = F.silu(x_in)
        x_in = x_in.transpose(1, 2)  # [B, L, d_inner]

        # SSM parametros dependentes do input
        x_dbl = self.x_proj(x_in)  # [B, L, d_state*2 + d_inner]
        delta, B_p, C_p = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner]

        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        # discretizar (Zero-Order Hold simplificado)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)              # [B, L, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B_p.unsqueeze(2)          # [B, L, d_inner, d_state]

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * x_in[:, t].unsqueeze(-1)
            y_t = (h * C_p[:, t].unsqueeze(1)).sum(-1)  # [B, d_inner]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # [B, L, d_inner]
        y = y + x_in * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class _MambaNet(nn.Module):
    def __init__(self, in_features: int, d_model: int = 64, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.blocks = nn.ModuleList(
            [_MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        for blk, ln in zip(self.blocks, self.norms):
            h = h + blk(ln(h))
        return self.head(h[:, -1, :])


class Mamba(ForecastModel):
    name = "mamba"
    input_kind = "sequence"

    def __init__(self, n_features: int, d_model: int = 64, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 epochs: int = 300, lr: float = 1e-3, batch_size: int = 64,
                 device: str = "cpu"):
        self.n_features = n_features
        self.d_model, self.n_layers, self.d_state = d_model, n_layers, d_state
        self.d_conv, self.expand = d_conv, expand
        self.epochs, self.lr, self.batch_size, self.device = epochs, lr, batch_size, device

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        self.net_ = _MambaNet(self.n_features, self.d_model, self.n_layers,
                              self.d_state, self.d_conv, self.expand)
        self.net_ = _train_loop(
            self.net_, X, y, X_val, y_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size, device=self.device,
        )
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net_(x).squeeze(-1).cpu().numpy()
