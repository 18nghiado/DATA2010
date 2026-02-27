
import torch
from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd

@dataclass
class TimeSeriesDataset:
    time: torch.Tensor
    x : torch.Tensor
    y: torch.Tensor
    feat_map : dict
    dataset_name : str


    def split(self, time_value) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        mask = self.time < time_value
        return TimeSeriesDataset(
            time=self.time[mask],
            y=self.y[mask],
            x=self.x[mask],
            feat_map=self.feat_map,
            dataset_name=self.dataset_name
        ), TimeSeriesDataset(
            time=self.time[~mask],
            y=self.y[~mask],
            x=self.x[~mask],
            feat_map=self.feat_map,
            dataset_name=self.dataset_name
        )

    def split_by_ratio(self, train: float, val: float, test: float) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset', 'TimeSeriesDataset']:
        assert abs(train + val + test - 1.0) < 1e-6, "Ratios must sum to 1"
        
        n = len(self)
        train_end = round(n * train)
        val_end = train_end + round(n * val)

        return (
            TimeSeriesDataset(time=self.time[:train_end],x=self.x[:train_end],y=self.y[:train_end],feat_map=self.feat_map, dataset_name= self.dataset_name),
            TimeSeriesDataset(time=self.time[train_end:val_end], x=self.x[train_end:val_end], y=self.y[train_end:val_end], feat_map=self.feat_map, dataset_name=self.dataset_name),
            TimeSeriesDataset(time=self.time[val_end:],x=self.x[val_end:],y=self.y[val_end:],feat_map=self.feat_map, dataset_name= self.dataset_name),
        )


    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for i in range(len(self.time)):
            yield self.time[i], self.x[i]  ,self.y[i]

    def __len__(self) -> int:
        return self.time.shape[0]