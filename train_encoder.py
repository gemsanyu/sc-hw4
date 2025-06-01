import pathlib
import json
from typing import List, Dict, Tuple
import math

import torch
from torch.utils.data import Dataset, DataLoader

from miner_objects import WIDTH, HEIGHT, DIAG
from encoder import AutoEncoder


class MinerDataset(Dataset):
    def __init__(self):
        dataset_dir = pathlib.Path("datasets")
        self.data_list:List[Dict] = []

        # Load all JSON episode files
        for file in dataset_dir.glob("episode_*.json"):
            with open(file, "r") as f:
                episode_data = json.load(f)
                self.data_list.extend(episode_data)
        print(len(self.data_list))

    def __getitem__(self, index):
        data = self.data_list[index]
        ship_raw_data = data["ship"]
        ship_data = [ship_raw_data["x"]/WIDTH, ship_raw_data["y"]/HEIGHT, ship_raw_data["fuel"]/100, math.sin(ship_raw_data["angle"]), math.cos(ship_raw_data["angle"])]
        ship_data = torch.as_tensor(ship_data, dtype=torch.float32)
        asteroids_raw_data = data["asteroids"]
        asteroids_data = []
        for asteroid_raw_row in asteroids_raw_data:
            ad = [asteroid_raw_row["x"]/WIDTH, asteroid_raw_row["y"]/HEIGHT,asteroid_raw_row["speed_x"]/WIDTH,asteroid_raw_row["speed_y"]/HEIGHT,asteroid_raw_row["radius"]/DIAG]
            asteroids_data.append(ad)
        asteroids_data = torch.as_tensor(asteroids_data, dtype=torch.float32)
        minerals_raw_data = data["minerals"]
        minerals_data = []
        for mineral_raw_row in minerals_raw_data:
            md = [mineral_raw_row["x"]/WIDTH, mineral_raw_row["y"]/HEIGHT]
            minerals_data.append(md)
        minerals_data = torch.as_tensor(minerals_data, dtype=torch.float32)
        
        return ship_data, asteroids_data, minerals_data

    def __len__(self):
        return len(self.data_list)   

def collate_fn(batch):
    return batch

if __name__ == "__main__":
    dataset = MinerDataset()
    auto_encoder = AutoEncoder()
    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=3e-4)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    loss_func = torch.nn.MSELoss()
    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in dataloader:
            total_loss = 0
            for bi, bdata in enumerate(batch):
                ship_data, asteroids_data, minerals_data = bdata
                ship_data_pred, asteroids_data_pred, minerals_data_pred = auto_encoder(ship_data, asteroids_data, minerals_data)
                bdata_orig = torch.cat((ship_data.ravel(),asteroids_data.ravel(),minerals_data.ravel()))
                pred_data = torch.cat((ship_data_pred.ravel(),asteroids_data_pred.ravel(),minerals_data_pred.ravel()))
                loss = loss_func(bdata_orig, pred_data)
                total_loss += loss
            optimizer.zero_grad()
            total_loss /= len(batch)
            total_loss.backward()
            epoch_loss.append(total_loss.item())
            optimizer.step()
        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        print(epoch_loss)
        if epoch_loss < 5e-4:
            break
    torch.save(auto_encoder.encoder.state_dict(),"encoder.pth")
        