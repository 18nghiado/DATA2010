
import torch
from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd

@dataclass
class TimeSeriesDataset:
    r"""
    A dataclass representing a time series dataset with features and labels.

    Args:
        - `time` (torch.Tensor): 1D tensor of UNIX timestamps (in seconds)
        - `x` (torch.Tensor): 2D feature tensor of shape `(N, num_features)`
        - `y` (torch.Tensor): Label tensor of shape `(N, 1)`
        - `feat_map` (dict): Mapping from feature name to its column index in `x`
        - `dataset_name` (str): Name of the source dataset or CSV file
    """
    time: torch.Tensor
    x : torch.Tensor
    y: torch.Tensor
    feat_map : dict
    dataset_name : str


    def split(self, time_value) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        r"""
        Split the dataset into two parts based on a timestamp threshold.

        Args:
            - `time_value` (float): UNIX timestamp (in seconds) used as the split boundary.
              Samples strictly before this value go to the first split.

        Returns: Tuple of two `TimeSeriesDataset`:
            - `before` : all samples where `time < time_value`
            - `after`  : all samples where `time >= time_value`
        """
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
        r"""
        Split the dataset into train, validation, and test sets by ratio.

        Args:
            - `train` (float): Proportion of data for training (e.g. `0.7`)
            - `val`   (float): Proportion of data for validation (e.g. `0.15`)
            - `test`  (float): Proportion of data for testing (e.g. `0.15`)

        Note: `train + val + test` must sum to `1.0`

        Returns: Tuple of three `TimeSeriesDataset`: `(train_data, val_data, test_data)`
        """
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
        r"""
        Iterate over samples in the dataset.

        Yields: Tuple of `(time_i, x_i, y_i)` for each sample `i`:
            - `time_i` (torch.Tensor): scalar timestamp
            - `x_i`    (torch.Tensor): 1D feature vector of shape `(num_features,)`
            - `y_i`    (torch.Tensor): label of shape `(1,)`
        """
        for i in range(len(self.time)):
            yield self.time[i], self.x[i]  ,self.y[i]

    def __len__(self) -> int:
        r"""
        Returns: Total number of samples `N` in the dataset.
        """
        return self.time.shape[0]