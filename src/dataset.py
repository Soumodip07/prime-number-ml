import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PrimeDataset(Dataset):

    def __init__(
        self,
        csv_file,
        mode="binary",          # "binary", "binary_mod", "mod"
        model_type="mlp"        # "mlp", "cnn", "transformer"
    ):

        df = pd.read_csv(csv_file, dtype={"binary": str})

        # -------- BINARY --------
        self.binary = np.array(
            [[int(b) for b in s] for s in df["binary"].values],
            dtype=np.float32
        )

        # -------- MOD FEATURES --------
        self.mods = df[
            ["mod2","mod3","mod5","mod7","mod11",
             "mod13","mod17","mod19","mod23","mod29","mod31"]
        ].values.astype(np.float32)

        # -------- LABELS --------
        self.labels = df["is_prime"].values.astype(np.float32)

        # -------- MODE CONTROL --------
        if mode == "binary":
            self.features = self.binary

        elif mode == "binary_mod":
            self.features = np.concatenate([self.binary, self.mods], axis=1)

        elif mode == "mod":
            self.features = self.mods

        else:
            raise ValueError("mode must be 'binary', 'binary_mod', or 'mod'")

        # -------- MODEL TYPE CONTROL --------
        self.model_type = model_type

        if model_type not in ["mlp", "cnn", "transformer"]:
            raise ValueError("model_type must be 'mlp', 'cnn', or 'transformer'")

        # Restrictions
        if model_type in ["cnn", "transformer"] and mode != "binary":
            raise ValueError("CNN/Transformer only supported for binary mode")

        self.input_dim = self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        x = torch.from_numpy(self.features[idx])

        # -------- MODEL-SPECIFIC SHAPES --------
        if self.model_type == "cnn":
            x = x.unsqueeze(0)  # (1, length)

        if self.model_type == "transformer":
            x = x.long()  # token-like input

        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x, y