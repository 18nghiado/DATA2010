
import torch
from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd

@dataclass
class TimeSeriesDataset:
    time: torch.Tensor
    x : torch.Tensor
    y: torch.Tensor


    def split(self, time_value) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        mask = self.time < time_value
        return TimeSeriesDataset(
            time=self.time[mask],
            y=self.y[mask],
            x=self.x[mask]
        ), TimeSeriesDataset(
            time=self.time[~mask],
            y=self.y[~mask],
            x=self.x[~mask]
        )


    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for i in range(len(self.time)):
            yield self.time[i], self.x[i]  ,self.y[i]

    def __len__(self) -> int:
        return self.time.shape[0]