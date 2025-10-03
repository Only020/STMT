import datetime
import json
import os
import random
import torch
import numpy as np
import argparse
import time

from tqdm import tqdm
import util
from util import log_string
from engine import trainer


# def parse_args(dataset):
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int,default=0,help='gpu index')
parser.add_argument('--k',type=int,default=15,help='k')
parser.add_argument('--l',type=int,default=2,help='l')
parser.add_argument('--dataset', type = str, default = 'pems08')
parser.add_argument('--data',type=str,default='data/pems08',help='data path')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--steps_per_day',type=int,default=288,help='')
parser.add_argument('--nhid',type=int,default=128,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default=None,help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--load_path', type = str, default = None)
parser.add_argument('--patience', type = int, default = 15)
parser.add_argument('--lr_mul', type = float, default = 1)
parser.add_argument('--n_warmup_steps', type = int, default = 4000)
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument("--clip", type=float, default=None, help='gradient clipping')
parser.add_argument("--gcn_depth", type=int, default=1, help='number of gcn layers[1,2,3]')
parser.add_argument("--propalpha", type=float, default=0.05, help='prop alpha[0.05,0.1,0.2,0.3]')
parser.add_argument("--alpha", type=float, default=3, help='alpha[1,2,3,4,5]')
args = parser.parse_args()
args = parser.parse_args()
with open(f"config/{args.dataset}.json", "r") as f:
    config = json.load(f)
args.__dict__.update(config)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

current_time = datetime.datetime.now()

formatted_time = current_time.strftime(r'%m_%d_%H_%M')
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        


def main():
    
    torch.cuda.set_device(args.device)
    
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")   
    
    log_path = f'./study/logs/{args.dataset}/'
    exp_path = f'./study/experiment/{args.dataset}/{formatted_time}/'
        
    create_directory_if_not_exists(log_path)
    create_directory_if_not_exists(exp_path)
        
    
    args.save = exp_path
    log = open(log_path+f'{formatted_time}.log', 'w')
    args.log = log_path+f'{formatted_time}.log'
    
    if args.seed != -1:

        log_string(log, "Start Deterministic Training with seed {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    #load data
    device = torch.device(args.device)
    
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,log)
    scaler = dataloader['scaler']

    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')


    engine = trainer(args.k,args.l,scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         device, args.gcn_depth,args.propalpha,args.alpha,
                         args.lr_mul, args.n_warmup_steps,args.lr,args.clip)

    log_string(log, "Train the model with {} parameters".format(count_parameters(engine.model)))


    log_string(log, "start training...")
    his_loss =[]
    val_time = []
    train_time = []
    wait = 0
    patience = args.patience
    best = 1e9
    for i in tqdm(range(1, args.epochs + 1), desc="Training Progress"):
        train_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:], i)
            train_loss.append(metrics[0])
            
        t2 = time.time()
        train_time.append(t2-t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_mae = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_mae.append(metrics[3])

        s2 = time.time()

        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mae = np.mean(valid_mae)
        his_loss.append(mvalid_loss)
        log_string(log, "Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f},Valid MAE: {:.4f},Training Time: {:.4f}/epoch".format(i, mtrain_loss, mvalid_loss,mvalid_rmse,mvalid_mae, (t2 - t1)))
        
        if best > his_loss[-1]:
            best = his_loss[-1]
            wait = 0
            torch.save(engine.model.state_dict(),args.save+f'epoch_{i}_{round(mvalid_loss,2)}.pth')
        else:
            wait = wait + 1
        if wait > patience:
            log_string(log, "Early Termination!")
            break
    log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+f'epoch_{bestid+1}_{round(his_loss[bestid],2)}.pth'))    


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    log_string(log, "Training finished")
    log_string(log, f"The best model is saved at epoch {bestid+1}, with valid loss {str(round(his_loss[bestid],4))}")

    amae = []
    amape = []
    armse = []
    results = {'prediction': [], 'ground_truth':[]}
    from copy import deepcopy as cp
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        results['prediction'].append(cp(pred).cpu().numpy())
        results['ground_truth'].append(cp(real).cpu().numpy())
        metrics = util.metric(pred,real)
        if i + 1 in [3, 6, 12]:
            log_string(log, f"Evaluate best model on test data for horizon {i+1}, Test MAE: {metrics[0]:.4f}, Test MAPE: {metrics[1]:.4f}, Test RMSE: {metrics[2]:.4f}")
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log_string(log, f"On average over 12 horizons, Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}")
    results['prediction'] = np.asarray(results['prediction'])
    results['ground_truth'] = np.asarray(results['ground_truth'])
    np.savez(args.save+f'prediction.npz', **results)
    log.close()


if __name__ == "__main__":
    main()
