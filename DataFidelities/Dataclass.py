# import torch
# import numpy as np
# from torch.utils.data import Dataset

# class TrainDataset(Dataset):

#     def __init__(self, train_ipt:torch.Tensor, 
#                        train_gdt:torch.Tensor, 
#                        train_y:torch.Tensor,
#                 ):
          
#         super(TrainDataset, self).__init__()
#         self.train_ipt = train_ipt
#         self.train_gdt = train_gdt
#         self.train_y = train_y

#     def __len__(self):
#         return self.train_gdt.shape[0]

#     def __getitem__(self, item):
#         return self.train_ipt[item], self.train_gdt[item], self.train_y[item]

# class ValidDataset(Dataset):

#     def __init__(self, valid_ipt:torch.Tensor, 
#                        valid_gdt:torch.Tensor, 
#                        valid_y:torch.Tensor,
#                        ):
          
#         super(ValidDataset, self).__init__()
#         self.valid_ipt = valid_ipt
#         self.valid_gdt = valid_gdt
#         self.valid_y = valid_y
#     def __len__(self):
#         return self.valid_gdt.shape[0]

#     def __getitem__(self, item):
#         return self.valid_ipt[item], self.valid_gdt[item], self.valid_y[item]

# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP


# def example(rank, world_size):
#     print(rank)
#     # create default process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
#     # create local model
#     model = nn.Linear(10, 10).to(rank)
#     # construct DDP model
#     ddp_model = DDP(model, device_ids=[rank])
#     # define loss function and optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     # forward pass
#     outputs = ddp_model(torch.randn(20, 10).to(rank))
#     labels = torch.randn(20, 10).to(rank)
#     # backward pass
#     loss_fn(outputs, labels).backward()
#     # update parameters
#     optimizer.step()
#     print('ss')

# def main():
#     world_size = 4
#     mp.spawn(example,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True)

# if __name__=="__main__":
#     main()


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

torch.manual_seed(40)

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


class Trainset(Dataset):
    def __init__(self, gdt:torch.Tensor, ipt:torch.Tensor):
        super(Trainset,self).__init__()
        self.gdt = gdt
        self.ipt = ipt
    
    def __len__(self):
        return self.gdt.shape[0]

    def __getitem__(self, item):
        return self.gdt[item], self.ipt[item]    

class Validset(Dataset):
    def __init__(self, gdt:torch.Tensor, ipt:torch.Tensor):
        super(Validset, self).__init__()
    
        self.gdt = gdt
        self.ipt = ipt
    
    def __len__(self):
        return self.gdt.shape[0]

    def __getitem__(self, item):
        return self.gdt[item], self.ipt[item]



def Basicblock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, name='layer'):
    
    return nn.Sequential(
        OrderedDict(
            [ (name + 'conv1',
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)
                ),
                (name + 'norm1', nn.BatchNorm2d(out_channels)),
                (name + 'act1', nn.ReLU()),
                (name + 'conv2',
                nn.Conv2d(
                    in_channels = out_channels,
                    out_channels = out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)
                ),
                (name + 'norm2', nn.BatchNorm2d(out_channels)),
                (name + 'act2', nn.ReLU())
            ]
        )
    )

class Resblock(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(Resblock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )
    def forward(self, x):
        return x + self.layers(x)

class DnCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, depth=12, num_features = 64, kernel_size=3, stride=1):
        super(DnCNN, self).__init__()
        padding = kernel_size//2
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channel, out_channels=num_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
        layers.append(nn.ReLU())

        for k in range(depth):
            layers.append(Basicblock(in_channels=num_features, out_channels=num_features, name='Layer%d'%(k)))#Resblock(n_channels=num_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))

        # for _ in range(depth):
        #     layers.append(nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        #     layers.append(nn.BatchNorm2d(num_features=num_features))
        #     layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=num_features, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding,bias=True))

        self.dnn = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x + self.dnn(x)
        return x

def epoch(network:nn.Module, dataloader:Dataset, optimizer=None, scheduler=None):

    average_loss = 0
    count = 0

    network.eval() if optimizer is None else network.train()

    for batch_gdt, batch_ipt in tqdm(dataloader):
        batch_gdt = batch_gdt.to(device)
        batch_ipt = batch_ipt.to(device)
        
        if optimizer:
            
            optimizer.zero_grad()
  
            batch_pre = network(batch_ipt)
            loss = torch.mean(torch.pow(batch_gdt - batch_pre, 2))
            loss.backward()
            torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1e-3)
            scheduler.step()
            optimizer.step()
            average_loss += loss.item()
        else:
            with torch.no_grad():
                batch_pre = network(batch_ipt)
                loss = torch.mean(torch.pow(batch_gdt - batch_pre, 2))
                average_loss += loss.item()
        
        count += 1
    return average_loss/len(dataloader.dataset) 

if __name__ == '__main__':

    gdt = torch.rand([100,1,64,64])
    ipt = torch.rand([100,1,64,64])

    trainset = Trainset(gdt,ipt)
    train_dataloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    gdt = torch.rand([100,1,64,64])
    ipt = torch.rand([100,1,64,64])

    vaildset = Validset(gdt,ipt)
    valid_dataloader = DataLoader(vaildset, batch_size=4, shuffle=True, num_workers=4)
    
    dncnn = DnCNN()
    network = nn.DataParallel(dncnn, device_ids=[0]).to(device)
    print(network)
    L2 = nn.MSELoss().to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=2e-7)
     
    max_epoches = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoches*len(train_dataloader), eta_min=1e-6)
    
    for i in range(max_epoches):

       average_loss = epoch(network, train_dataloader, optimizer, scheduler)
       print('Epoch = ', i, 'MSELoss = ', average_loss)

    

























            
