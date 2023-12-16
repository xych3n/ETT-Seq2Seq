from typing import Literal, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, filename: str, split: Literal["train", "test", "val"],
                 in_size: int, out_size: int, seed: Optional[int] = None) -> None:
        samples = []
        with open(filename) as f:
            rows = iter(f)
            next(rows)
            for row in rows:
                cols = row.split(",")
                features = list(map(float, cols[1:]))
                samples.append(features)
        # Normalize
        samples = minmax_scale(samples)
        # Sample with Sliding Window
        inputs = list(zip(*[samples[i:] for i in range(in_size)]))
        targets = list(zip(*[samples[i+in_size:] for i in range(out_size)]))
        inputs = inputs[:len(targets)]
        # Split Dataset
        n = len(inputs)
        train_size = int(n * 0.6)
        test_size = int(n * 0.2)
        val_size = n - train_size - test_size
        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_size+val_size, train_size=train_size, random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=test_size, train_size=val_size, random_state=seed)
        if split == "train":
            self.inputs = np.array(X_train, dtype=np.float32)
            self.targets = np.array(y_train, dtype=np.float32)
        elif split == "test":
            self.inputs = np.array(X_test, dtype=np.float32)
            self.targets = np.array(y_test, dtype=np.float32)
        elif split == "val":
            self.inputs = np.array(X_val, dtype=np.float32)
            self.targets = np.array(y_val, dtype=np.float32)
        else:
            raise ValueError("split should be one of 'train', 'test', and 'val', but got {!r}".format(split))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]
