import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from utils import *
from model import SIMBAD
from data_preparation import *
from time import time
from metric import *

args = argparse.ArgumentParser(description='SIMBAD Model')

args.add_argument('--device', default='cuda', type=str, help='use GPU')
args.add_argument('--mode', default='train', type=str, help='choose mode train or test')

# Data prepare
args.add_argument('--data_dir', default='data', type=str, help='dataset directory')
args.add_argument('--data_type', default='PEMS08', type=str, help='type of traffic data')

# Model details
args.add_argument('--weeks', default=2, type=int, help='number of weeks')
args.add_argument('--days', default=1, type=int, help='number of days')
args.add_argument('--hours', default=1, type=int, help='number of hours')
args.add_argument('--points_per_hour', default=12, type=int, help='number of points per hour')
args.add_argument('--window', default=6, type=int, help='window size for similarity comparison')
args.add_argument('--input_dim', default=1, type=int, help='number of input dimensions')
args.add_argument('--hidden_dim', default=64, type=int, help='number of hidden dimensions')
args.add_argument('--output_dim', default=1, type=int, help='number output dimensions')
args.add_argument('--num_vertices', default=170, type=int, help='number of vertices')
args.add_argument('--seq_len', default=12, type=int, help='input and output sequence length')
args.add_argument('--train_ratio', default=0.6, type=float, help='training dataset ratio')
args.add_argument('--val_ratio', default=0.2, type=float, help='validation dataset ratio')
args.add_argument('--dropout', default=0.2, type=float, help='dropout ratio')
args.add_argument('--scale', default=20, type=int, help='scaling value for temporal threshold')

# Training details
args.add_argument('--batch_size', default=64, type=int)
args.add_argument('--epochs', default=100, type=int, help='max number of epochs for training')
args.add_argument('--lr_rate', default=0.001, type=float)
args.add_argument('--early_stop', default=20, type=int, help='number of patience to wait of validation loss improval')
args.add_argument('--metric', default='mae', type=str, help='loss calculation metric')
args.add_argument('--print_every', default=200, type=int, help='print training process')
args.add_argument('--checkpoint', default='PEMS08.pth', type=str, help='name of saved model')
args.add_argument('--set_seed', default=True, type=bool)
args.add_argument('--seed', type=int, default=40)

args = args.parse_args()


def load_data(args, dataset_sequence):
    all_data = read_and_generate_dataset(dataset_sequence, args.weeks, args.days, args.hours, args.seq_len, args.window, args.input_dim, 
                                        args.train_ratio, args.val_ratio, args.points_per_hour)

    train_loader, val_loader, test_loader, stats = get_final_dataset(all_data, args.batch_size)

    return train_loader, val_loader, test_loader, stats

def get_weighted_adjacency(device, week, day, hour, stats):

    week_min_val = torch.from_numpy(stats['week']['min'][...,:1])
    week_max_val = torch.from_numpy(stats['week']['max'][...,:1])
    day_min_val = torch.from_numpy(stats['day']['min'][...,:1])
    day_max_val = torch.from_numpy(stats['day']['max'][...,:1])
    hour_min_val = torch.from_numpy(stats['hour']['min'][...,:1])
    hour_max_val = torch.from_numpy(stats['hour']['max'][...,:1])

    def normalize(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-10)
    
    week = normalize(week, week_min_val, week_max_val).to(device)
    day = normalize(day, day_min_val, day_max_val).to(device)
    hour = normalize(hour, hour_min_val, hour_max_val).to(device)
    
    week_adj = get_sample_matrix(week, week)
    day_adj = get_sample_matrix(day, day)
    hour_adj = get_sample_matrix(hour, hour)

    return week, day, hour, week_adj, day_adj, hour_adj

def compute_val_loss(device, model, val_loader, loss_func, epoch, stats):
    '''
    :param:
    model : model
    val_loader : validation dataset
    loss_func : loss metric
    epoch : epoch

    :return:
    loss : validation loss
    '''
    model.eval()
    with torch.no_grad():
        y_true_list = []
        y_pred_list = []

        for index, (w_train, d_train, h_train, week2_sim, week_sim, day_sim, target) in enumerate(val_loader):
            week, day, hour, week_adj, day_adj, hour_adj = get_weighted_adjacency(device, w_train, d_train, h_train, stats)

            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)
            week_adj = week_adj.to(device)
            day_adj = day_adj.to(device)
            hour_adj = hour_adj.to(device)

            pred = model([week, day, hour, week2_sim, week_sim, day_sim, week_adj, day_adj, hour_adj])

            y_true_list.append(target.detach().cpu())
            y_pred_list.append(pred.detach().cpu())

        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)

        loss = loss_func(y_pred, y_true)

    print('Epoch: %s, validation loss: %.2f' % (epoch, loss))

    return loss

def train(device, args, model, train_loader, val_loader, optimizer, loss_func, stats):
    global_step = 0
    best_epoch = 0
    best_val_loss = float('inf')
    early_stop = 0
    start_time = time()
    

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch+1}')

        model.train()

        for index, (week, day, hour, week2_sim, week_sim, day_sim, target) in enumerate(train_loader):
            optimizer.zero_grad()

            week, day, hour, week_adj, day_adj, hour_adj = get_weighted_adjacency(device, week, day, hour, stats)

            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)
            week_adj = week_adj.to(device)
            day_adj = day_adj.to(device)
            hour_adj = hour_adj.to(device)

            outputs = model([week, day, hour, week2_sim, week_sim, day_sim, week_adj, day_adj, hour_adj])
            loss = loss_func(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1

            if global_step % args.print_every == 0:
                print('Global step: %s, Training loss: %.2f, Time: %.2fs' % (global_step, training_loss, time() - start_time))

        val_loss = compute_val_loss(device, model, val_loader, loss_func, epoch+1, stats)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            print(f"Best validation loss: {best_val_loss:.2f} at epoch {best_epoch}")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'loss': best_val_loss,
                'global_step': global_step,
                'time_taken' : time() - start_time,
                'seed' : args.seed
                }, args.checkpoint)
            early_stop = 0
        else:
            early_stop += 1
        
        if early_stop == args.early_stop:
            print(f"Performance did not improve for {args.early_stop} steps. End training.")
            break

    end_time = time()
    end_min = int((end_time - start_time)//60)
    end_sec = int((end_time - start_time)%60)

    print(f"Training took {end_min} minutes {end_sec} seconds.")

def train_continue(device, args, model, train_loader, val_loader, optimizer, loss_func, epoch_saved, val_loss, stats):
    global_step = 0
    best_epoch = 0
    best_val_loss = val_loss
    early_stop = 0
    start_time = time()
    

    for epoch in range(epoch_saved, args.epochs):
        print(f'Epoch: {epoch}')

        model.train()

        for index, (week, day, hour, week2_sim, week_sim, day_sim, target) in enumerate(train_loader):
            optimizer.zero_grad()

            week, day, hour, week_adj, day_adj, hour_adj = get_weighted_adjacency(device, week, day, hour, stats)

            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)
            week_adj = week_adj.to(device)
            day_adj = day_adj.to(device)
            hour_adj = hour_adj.to(device)

            outputs = model([week, day, hour, week2_sim, week_sim, day_sim, week_adj, day_adj, hour_adj])
            loss = loss_func(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1

            if global_step % args.print_every == 0:
                print('Global step: %s, Training loss: %.2f, Time: %.2fs' % (global_step, training_loss, time() - start_time))

        val_loss = compute_val_loss(device, model, val_loader, loss_func, epoch+1, stats)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            print(f"Best validation loss: {best_val_loss:.2f} at epoch {best_epoch}")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'loss': best_val_loss,
                'global_step': global_step,
                'time_taken' : time() - start_time,
                'seed' : args.seed
                }, args.checkpoint)
            early_stop = 0
        else:
            early_stop += 1
        
        if early_stop == args.early_stop:
            print(f"Performance did not improve for {args.early_stop} steps. End training.")
            break

    end_time = time()
    end_min = int((end_time - start_time)//60)
    end_sec = int((end_time - start_time)%60)

    print(f"Training took {end_min} minutes {end_sec} seconds.")

def test(device, model, test_loader, stats):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for index, (week, day, hour, week2_sim, week_sim, day_sim, target) in enumerate(test_loader):
            week, day, hour, week_adj, day_adj, hour_adj = get_weighted_adjacency(device, week, day, hour, stats)

            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)
            week_adj = week_adj.to(device)
            day_adj = day_adj.to(device)
            hour_adj = hour_adj.to(device)

            output = model([week, day, hour, week2_sim, week_sim, day_sim, week_adj, day_adj, hour_adj])
            test_t = target.permute(0,2,1).unsqueeze(-1)
            output = output.permute(0,2,1).unsqueeze(-1)
            y_true.append(test_t)
            y_pred.append(output)
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    rmse, mae, mape = RMSE_MAE_MAPE(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")


def main(args, device):
    if args.set_seed:
        def init_seed(seed):
            '''
            Disable cudnn to maximize reproducibility
            '''
            torch.cuda.cudnn_enabled = False
            torch.backends.cudnn.deterministic = True
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    init_seed(args.seed)

    dataset_sequence = f"{args.data_dir}/{args.data_type}/{args.data_type}.npz"
    adj_filename = f"{args.data_dir}/{args.data_type}/{args.data_type}.csv"

    adj = torch.from_numpy(get_adjacency_matrix(adj_filename, args.num_vertices)).unsqueeze(0).to(args.device)

    train_loader, val_loader, test_loader, stats = load_data(args, dataset_sequence)

    backbone = get_backbones(adj_filename, args.num_vertices, args.input_dim, args.hidden_dim)

    model = SIMBAD(device, args.seq_len, backbone, adj, args.num_vertices, args.input_dim, args.output_dim, args.dropout, args.scale)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)

    criterion = MaskedMAELoss()
    
    if args.mode == 'train':
        train(device, args, model, train_loader, val_loader, optimizer, criterion, stats)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Saved model succesfully loaded')
        test(device, model, test_loader, stats)
    elif args.mode == 'train_continue':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        epoch_continue = checkpoint["epoch"]
        val_loss = checkpoint['loss']
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model successfully loaded. Continue training from epoch {checkpoint['epoch']}.")
        train_continue(device, args, model, train_loader, val_loader, test_loader, optimizer, criterion, epoch_continue, val_loss, stats)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Saved model successfully loaded')
        test(device, model, test_loader, stats)
    elif args.mode == 'test':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Saved model successfully loaded')
        test(device, model, test_loader, stats)

if __name__ == '__main__':
    device = torch.device(args.device)
    main(args, device)
