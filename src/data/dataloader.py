import pandas as pd
import torch 
import os
import numpy as np
from typing import Tuple, List

from .timeseries_data import TimeSeriesDataset


class Dataloader:
    def __init__(self, path):
        if path is None:
            raise ValueError("Please specify path")
        self.path = path

    def _normalize(pd_series : pd.Series)-> pd.Series:
        series_max = pd_series.max()
        series_min = pd_series.min()

        return (pd_series - series_min)/(series_max - series_min)
        
    def from_csv(
            self,
            csv_file_name:str,
            feat_columns : None | List[str] = [],

    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        if not os.path.isfile(os.path.join(self.path, csv_file_name)):
            raise FileExistsError(f"{csv_file_name} doesn't not exist in {self.path}")
        
        
        df = pd.read_csv(os.path.join(self.path, csv_file_name))
        
        missing_columns = [col for col in feat_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in dataframe: {missing_columns}")
        

        df['Date'] = pd.to_datetime(df['Date'])

        feats = torch.from_numpy(df[feat_columns].to_numpy())


        time = torch.from_numpy(df['Date'].astype(np.int64).to_numpy())
        binary_labels = torch.from_numpy(df['price_increase'].to_numpy())
        regression_labels = torch.from_numpy(df['next_close'].to_numpy())
        return TimeSeriesDataset(
            time=time,
            binary_y=binary_labels,
            regression_y=regression_labels,
            x = feats
        )

