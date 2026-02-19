
import torch
from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd

@dataclass
class TimeSeriesDataset:
    time: torch.Tensor
    x : torch.Tensor
    binary_y: torch.Tensor
    regression_y: torch.Tensor


    def split(self, time_value) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        mask = self.time < time_value
        return TimeSeriesDataset(
            time=self.time[mask],
            binary_y=self.binary_y[mask],
            regression_y=self.regression_y[mask],
            x=self.x[mask]
        ), TimeSeriesDataset(
            time=self.time[~mask],
            binary_y=self.binary_y[~mask],
            regression_y=self.regression_y[~mask],
            x=self.x[~mask]
        )


    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for i in range(len(self.time)):
            yield self.time[i], self.x[i]  ,self.binary_y[i], self.regression_y[i] 

    def __len__(self) -> int:
        return self.time.shape[0]