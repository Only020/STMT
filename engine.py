import torch
import torch.nn as nn
import torch.optim as optim
from model import STMT
# from models.model import TESTAM
import util

class trainer():
    def __init__(self,k,l, scaler, in_dim, seq_length, num_nodes, nhid, dropout, device,gcn_depth=1,propalpha=0.05,alpha=3, lr_mul = 1.,
                 n_warmup_steps = 2000,lrate = 0.001,clip=None
                 ):
        self.model = STMT(device,k, num_nodes,gcn_depth,propalpha,alpha,
                           dropout,in_dim=in_dim, out_dim=seq_length, hidden_size=nhid,layers=l)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lrate, betas = (0.9, 0.98), eps = 1e-9)
        self.schedule = util.CosineWarmupScheduler(self.optimizer, d_model = nhid, n_warmup_steps = n_warmup_steps, lr_mul = lr_mul)
        self.criterion = util.masked_mae
        self.scaler = scaler
        self.clip = clip
        self.cur_epoch = 0
    
    def train(self, input, real_val, cur_epoch):
        self.model.train()
        self.schedule.zero_grad()
        
        output = self.model(input)
        
        predict = self.scaler.inverse_transform(output)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        real = real.squeeze(dim = 1).unsqueeze(dim = -1)
        loss = self.criterion(predict, real)#+self.model.minmax_loss[0]+self.model.minmax_loss[1]
        # loss = self.criterion(predict, real)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.schedule.step_and_update_lr()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        mae = util.masked_mae(predict, real, 0.0).item()
        return loss.item(),mape,rmse,mae
    
    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        real = real.squeeze(dim = 1).unsqueeze(dim = -1)
        predict = self.scaler.inverse_transform(output)
        loss = self.criterion(predict, real)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        mae = util.masked_mae(predict,real,0.0).item()
        return loss.item(),mape,rmse,mae
