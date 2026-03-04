import pandas as pd
import torch 
import os
import numpy as np
from typing import Tuple, List

from .timeseries_data import TimeSeriesDataset

"""
Load raw data to memory

"""
class Dataloader:
    """r
    Load raw data into memory and return `TimeSeriesDataset` for each data.

    Args:
        - `path`(str): path to directory that raw data is stored
    
    """
    def __init__(self, path: str):
        if path is None:
            raise ValueError("Please specify path")
        self.path = path
        
    def from_csv(
            self,
            csv_file_name:str,
            label_column,
            feat_columns : None | List[str] = [],
    ) -> TimeSeriesDataset:
        r"""
        Load raw data from CSV into memory and return `TimeSeriesDataset` for each data

        Args:
            - `csv_file_name` (str): Name of the file that need to load
            - `label_column`(str): which column is considered as label vector
            - `feat_columns` (List): List of features columns

        Return: `TimeSeriesDataset` dataset
        """
        if not os.path.isfile(os.path.join(self.path, csv_file_name)):
            raise FileExistsError(f"{csv_file_name} doesn't not exist in {self.path}")
        
        
        df = pd.read_csv(os.path.join(self.path, csv_file_name))
        df = df.iloc[:-1]
        
        missing_columns = [col for col in feat_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in dataframe: {missing_columns}")
        

        df['Date'] = pd.to_datetime(df['Date'])

        df = df[df['Date'] <= '2023-12-19'] # Remove this when preprocess data is updated


        feats = torch.from_numpy(df[feat_columns].to_numpy()).float()

        feat_map = {feat: idx for idx, feat in enumerate(feat_columns)}


        time = torch.from_numpy(df['Date'].astype(np.int64).to_numpy()/1_000_000_000)
        labels = torch.from_numpy(df[label_column].to_numpy()).unsqueeze(-1)
        return TimeSeriesDataset(
            time=time,
            y=labels,
            x = feats,
            feat_map= feat_map,
            dataset_name=csv_file_name
        )

