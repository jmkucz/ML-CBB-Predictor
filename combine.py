from schedule_getter import schedule
import random
from data_getter import data
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_data():
    home = "Michigan"
    away = "Kent State"

    sched = schedule(home, away)
    dat = data()
    results = []
    games = []
    for batch in sched:
        for game in batch:
            i = 0
            homeStats = []
            awayStats = []
            for team in game:
                if i == 0:
                    homeStats= dat[team]
                elif i == 1:
                    awayStats = dat[team]
                else:
                    results.append(np.array([team]))
                i = i + 1
            comb = []
            for i in range(len(homeStats)):
                comb.append(homeStats[i] - awayStats[i])
            games.append(comb)

    # shuffle games for separation
    temp = list(zip(games, results))
    random.shuffle(temp)
    shuffled_games, shuffled_results = zip(*temp)
    train_G = []
    train_R = []
    val_G = []
    val_R = []
    for i in range(len(shuffled_games)):
        if i / len(shuffled_games) < .7:
            train_G.append(shuffled_games[i])
            train_R.append(shuffled_results[i])
        else:
            val_G.append(shuffled_games[i])
            val_R.append(shuffled_results[i])
    
    tensor_x_train =  torch.Tensor(train_G)
    tensor_y_train = torch.Tensor(train_R)
    tensor_x_val = torch.Tensor(val_G)
    tensor_y_val = torch.Tensor(val_R)

    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    train_dataloader = DataLoader(train_dataset)

    val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
    val_dataloader = DataLoader(val_dataset)
    return train_dataloader, val_dataloader

    
            
