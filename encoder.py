from typing import Tuple

import torch
import torch.nn as nn

from graph_encoder import GraphAttentionEncoder


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ship_fc = nn.Linear(5, 32, bias=False)
        self.asteroid_fc = nn.Linear(5, 32, bias=False)
        self.mineral_fc = nn.Linear(2, 32, bias=False)
        self.gae = GraphAttentionEncoder(n_heads=8, embed_dim=32, n_layers=3)
        self.obj_proj = nn.Linear(32, 8, bias=False)
        self.graph_proj = nn.Linear(32, 8, bias=False)

    def forward(self,
                ship_data: torch.Tensor,
                asteroids_data: torch.Tensor,
                minerals_data: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        ship_data = ship_data.view(1,-1)
        ship_embed = self.ship_fc(ship_data)
        asteroids_embed = self.asteroid_fc(asteroids_data)
        minerals_embed = self.mineral_fc(minerals_data)
        raw_embeds = torch.concatenate((ship_embed, asteroids_embed, minerals_embed), dim=0)
        raw_embeds = raw_embeds[None, :, :]
        obj_embeds, graph_embeds = self.gae(raw_embeds)
        obj_embeds = obj_embeds[0]
        obj_embeds = self.obj_proj(obj_embeds)
        graph_embeds = graph_embeds[0]
        graph_embeds = self.graph_proj(graph_embeds)
        return obj_embeds, graph_embeds
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ship_fc = nn.Sequential(nn.Linear(16, 32), 
                                     nn.ReLU(),
                                     nn.Linear(32, 16), 
                                     nn.ReLU(), 
                                     nn.Linear(16,5))
        self.asteroid_fc = nn.Sequential(nn.Linear(16, 32), 
                                     nn.ReLU(),
                                     nn.Linear(32, 16), 
                                     nn.ReLU(), 
                                     nn.Linear(16,5))
        self.mineral_fc = nn.Sequential(nn.Linear(16, 32), 
                                     nn.ReLU(),
                                     nn.Linear(32, 16), 
                                     nn.ReLU(), 
                                     nn.Linear(16,2))
        
    def forward(self, 
                ship_embed:torch.Tensor,
                asteroids_embed: torch.Tensor,
                minerals_embed: torch.Tensor,
                graph_embed:torch.Tensor
                )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        ship_embed = torch.concatenate((ship_embed, graph_embed), dim=1)
        ship_data = self.ship_fc(ship_embed)
        asteroids_embed = torch.concatenate((asteroids_embed,torch.repeat_interleave(graph_embed, len(asteroids_embed), dim=0)), dim=1)
        asteroids_data = self.asteroid_fc(asteroids_embed)
        minerals_embed = torch.concatenate((minerals_embed, torch.repeat_interleave(graph_embed, len(minerals_embed), dim=0)), dim=1)
        minerals_data = self.mineral_fc((minerals_embed))
        return ship_data, asteroids_data, minerals_data
    
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, ship_data: torch.Tensor,
                asteroids_data: torch.Tensor,
                minerals_data: torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        num_asteroids = asteroids_data.shape[0]
        num_minerals = minerals_data.shape[0]
        obj_embeds, graph_embed = self.encoder(ship_data, asteroids_data, minerals_data)
        ship_embed = obj_embeds[[0], :]
        asteroids_embed = obj_embeds[1:num_asteroids+1,:]
        minerals_embed = obj_embeds[num_asteroids+1:,:]
        
        ship_data_pred, asteroids_data_pred, minerals_data_pred = self.decoder(ship_embed, 
                                                                               asteroids_embed,
                                                                               minerals_embed,
                                                                               graph_embed)
        return ship_data_pred, asteroids_data_pred, minerals_data_pred
        