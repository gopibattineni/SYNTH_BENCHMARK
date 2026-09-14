"""Generator adapters — wrap reference implementations without modifying them."""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..paths import CTAB_ROOT, DIFFUSION_MODULE, SYNTH_ROOT


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sdv_metadata(df: pd.DataFrame):
    from sdv.metadata import SingleTableMetadata

    md = SingleTableMetadata()
    md.detect_from_dataframe(df)
    return md


def train_sdv(name: str, train: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    from sdv.single_table import (
        CTGANSynthesizer,
        CopulaGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    set_global_seed(seed)
    metadata = _sdv_metadata(train)
    cls = {
        "CTGAN": CTGANSynthesizer,
        "CopulaGAN": CopulaGANSynthesizer,
        "TVAE": TVAESynthesizer,
        "GaussianCopula": GaussianCopulaSynthesizer,
    }[name]
    # SDV synthesizers accept epochs / seed via constructor where available
    kwargs: dict = {"metadata": metadata}
    try:
        synth = cls(**kwargs)
    except TypeError:
        synth = cls(metadata)
    # Prefer setting seed if API supports it
    if hasattr(synth, "set_random_state"):
        synth.set_random_state(seed)
    synth.fit(train)
    return synth.sample(num_rows=n_samples)


def train_ctabgan(train: pd.DataFrame, target: str, n_samples: int, seed: int, task: str) -> pd.DataFrame:
    set_global_seed(seed)
    if not CTAB_ROOT.is_dir():
        raise FileNotFoundError(f"CTAB-GAN-Plus not found at {CTAB_ROOT}")
    sys.path.insert(0, str(CTAB_ROOT))
    from model.ctabgan import CTABGAN  # type: ignore

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "train.csv"
        train.to_csv(csv_path, index=False)
        problem = {"Classification": target} if task == "classification" else {"Regression": target}
        cat_cols = [c for c in train.columns if train[c].dtype == object or str(train[c].dtype) == "category"]
        if target not in cat_cols and task == "classification":
            cat_cols.append(target)
        model = CTABGAN(
            raw_csv_path=str(csv_path),
            categorical_columns=cat_cols,
            log_columns=[],
            mixed_columns={},
            integer_columns=[],
            problem_type=problem,
        )
        model.fit()
        synth = model.data_prep.inverse_prep(model.synthesizer.sample(n_samples))
    return pd.DataFrame(synth)


def train_wgan_gp(
    train: pd.DataFrame,
    target: str,
    n_samples: int,
    seed: int,
    epochs: int = 100,
    task: str = "classification",
) -> pd.DataFrame:
    """Compact WGAN-GP matching cancer.py Other-GANS pattern (train on train only)."""
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    set_global_seed(seed)
    data = train.copy()
    encoder = None
    # Always encode classification targets (including numeric 0/1 Group labels)
    if task == "classification" or data[target].dtype == object or str(data[target].dtype) == "category":
        encoder = LabelEncoder()
        data[target] = encoder.fit_transform(data[target].astype(str))

    # Encode remaining object columns
    encoders = {}
    for col in data.columns:
        if col == target:
            continue
        if data[col].dtype == object or str(data[col].dtype) == "category":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.astype(float))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_tensor = torch.tensor(scaled, dtype=torch.float32)
    batch_size = min(64, len(real_tensor))
    latent_dim = 64
    data_dim = real_tensor.shape[1]
    loader = torch.utils.data.DataLoader(real_tensor, batch_size=batch_size, shuffle=True)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, data_dim),
            )

        def forward(self, z):
            return self.model(z)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(data_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.model(x)

    G = Generator().to(device)
    C = Critic().to(device)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))

    def gradient_penalty(real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, device=device).expand_as(real_samples)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        crit = C(interpolates)
        grads = torch.autograd.grad(
            outputs=crit,
            inputs=interpolates,
            grad_outputs=torch.ones_like(crit),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads = grads.view(grads.size(0), -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    for _ in range(epochs):
        for real_batch in loader:
            real_batch = real_batch.to(device)
            for _k in range(5):
                z = torch.randn(real_batch.size(0), latent_dim, device=device)
                fake_batch = G(z).detach()
                loss_C = C(fake_batch).mean() - C(real_batch).mean() + 10 * gradient_penalty(real_batch, fake_batch)
                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()
            z = torch.randn(real_batch.size(0), latent_dim, device=device)
            loss_G = -C(G(z)).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        synth_scaled = G(z).cpu().numpy()
    synth = pd.DataFrame(scaler.inverse_transform(synth_scaled), columns=data.columns)
    for col, le in encoders.items():
        vals = np.clip(np.rint(synth[col]), 0, len(le.classes_) - 1).astype(int)
        synth[col] = le.inverse_transform(vals)
    if encoder is not None:
        vals = np.clip(np.rint(synth[target]), 0, len(encoder.classes_) - 1).astype(int)
        synth[target] = encoder.inverse_transform(vals)
        # restore original dtype when labels were numeric classes (e.g. 0/1)
        if pd.api.types.is_numeric_dtype(train[target]):
            synth[target] = pd.to_numeric(synth[target], errors="coerce")
    return synth


def train_diffusion(name: str, train: pd.DataFrame, target: str, n_samples: int, seed: int, task: str) -> pd.DataFrame:
    set_global_seed(seed)
    # Import reference module without modifying it
    import importlib.util

    spec = importlib.util.spec_from_file_location("diffusion_generators_ref", DIFFUSION_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {DIFFUSION_MODULE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    is_reg = task == "regression"
    if name == "TabDDPM":
        return mod.train_tabddpm(train, target_col=target, n_samples=n_samples, seed=seed, is_regression=is_reg)
    if name == "ForestDiffusion":
        return mod.train_forestdiffusion(train, target_col=target, n_samples=n_samples, seed=seed, is_regression=is_reg)
    raise ValueError(name)


def generate_synthetic(
    generator: str,
    train: pd.DataFrame,
    target: str,
    n_samples: int,
    seed: int,
    task: str,
) -> pd.DataFrame:
    """Train generator on TRAIN only and return synthetic dataframe."""
    if generator in {"CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"}:
        return train_sdv(generator, train, n_samples, seed)
    if generator == "CTABGAN":
        return train_ctabgan(train, target, n_samples, seed, task)
    if generator == "WGAN_GP":
        return train_wgan_gp(train, target, n_samples, seed, task=task)
    if generator in {"TabDDPM", "ForestDiffusion"}:
        return train_diffusion(generator, train, target, n_samples, seed, task)
    raise ValueError(f"Unknown generator: {generator}")
