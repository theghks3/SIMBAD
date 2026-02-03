import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configparser
from utils import *
from model import *
from data_preparation import *
from time import time
from torch.utils.data import TensorDataset, DataLoader
from metric import *
from collections import OrderedDict


def load_data(args, adj_filename, dataset_sequence):
    all_data = read_and_generate_dataset(adj_filename, dataset_sequence, args.weeks, args.days, args.hours, args.num_nodes,
                                        args.seq_len, args.input_dim, args.train_ratio, args.val_ratio, args.points_per_hour, args.node_ratio)

    train_loader, val_loader, test_loader, stats = get_final_dataset(all_data, args.batch_size)

    return train_loader, val_loader, test_loader, stats, all_data['threshold']['week'], all_data['threshold']['day'], all_data['sem_adj']

def compute_val_loss(device, model, val_loader, scaler, loss_func, epoch):
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

        for index, (week, day, hour, week2_sim, week_sim, day_sim, target) in enumerate(val_loader):
            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)
            pred = model([week, day, hour, week2_sim, week_sim, day_sim])

            y_true_list.append(target.to(device).detach().cpu())
            y_pred_list.append(pred.detach().cpu())

        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        y_pred = scaler.inverse_transform(y_pred)

        loss = loss_func(y_pred, y_true)
    log_string(log, f'Epoch: {epoch}, validation loss: {loss:.2f}')

    return loss

def train(device, args, model, train_loader, val_loader, test_loader, scaler, optimizer, loss_func):
    global_step = 0
    best_epoch = 0
    best_val_loss = float('inf')
    early_stop = 0
    start_time = time()
    

    for epoch in range(args.epochs):
        log_string(log, f'Epoch: {epoch+1}')

        model.train()

        for index, (week, day, hour, week2_sim, week_sim, day_sim, target) in enumerate(train_loader):
            optimizer.zero_grad()

            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)

            outputs = model([week, day, hour, week2_sim, week_sim, day_sim])
            outputs = scaler.inverse_transform(outputs)
            loss = loss_func(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1

            if global_step % args.print_every == 0:
                log_string(log, f'Global step: {global_step}, Training loss: {training_loss:.2f}, Time: {time() - start_time:.2f}s')

        val_loss = compute_val_loss(device, model, val_loader, scaler, loss_func, epoch+1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            log_string(log, f'Best validation loss: {best_val_loss:.2f} at epoch {best_epoch}')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'loss': best_val_loss,
                'global_step': global_step,
                'time_taken' : time() - start_time
                }, args.checkpoint)
            early_stop = 0
        else:
            early_stop += 1
        
        if early_stop == args.early_stop:
            log_string(log, f"Performance did not improve for {args.early_stop} steps. End training.")
            break

    end_time = time()
    end_min = int((end_time - start_time)//60)
    end_sec = int((end_time - start_time)%60)

    log_string(log, f"Training took {end_min} minutes {end_sec} seconds.")

def train_continue(device, args, model, train_loader, val_loader, test_loader, scaler, optimizer, loss_func, epoch_saved, val_loss):
    global_step = 0
    best_epoch = 0
    best_val_loss = val_loss
    early_stop = 0
    start_time = time()
    

    for epoch in range(epoch_saved, args.epochs+1):
        log_string(log, f'Epoch: {epoch}')

        model.train()

        for index, (week, day, hour, week2_sim, week_sim, day_sim, target) in enumerate(train_loader):
            optimizer.zero_grad()

            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)

            outputs = model([week, day, hour, week2_sim, week_sim, day_sim])
            outputs = scaler.inverse_transform(outputs)
            loss = loss_func(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1

            if global_step % args.print_every == 0:
                log_string(log, f'Global step: {global_step}, Training loss: {training_loss:.2f}, Time: {time() - start_time:.2f}s')

        val_loss = compute_val_loss(device, model, val_loader, scaler, loss_func, epoch+1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            log_string(log, f"Best validation loss: {best_val_loss:.2f} at epoch {best_epoch}")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'loss': best_val_loss,
                'global_step': global_step,
                'time_taken' : time() - start_time
                }, args.checkpoint)
            early_stop = 0
        else:
            early_stop += 1
        
        if early_stop == args.early_stop:
            log_string(log, f"Performance did not improve for {args.early_stop} steps. End training.")
            break

    end_time = time()
    end_min = int((end_time - start_time)//60)
    end_sec = int((end_time - start_time)%60)

    log_string(log, f"Training took {end_min} minutes {end_sec} seconds.")

def test(device, args, model, test_loader, scaler, loss_func):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for index, (week, day, hour, week2_sim, week_sim, day_sim, target) in enumerate(test_loader):
            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            week2_sim = week2_sim.to(device)
            week_sim = week_sim.to(device)
            day_sim = day_sim.to(device)
            output = model([week, day, hour, week2_sim, week_sim, day_sim])
            test_t = target.to(device).permute(0,2,1).unsqueeze(-1)
            output = output.permute(0,2,1).unsqueeze(-1)
            y_true.append(test_t)
            y_pred.append(output)
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()[...,0]
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()[...,0]
    y_pred = scaler.inverse_transform(y_pred)

    for i in range(args.points_per_hour):
        real_point = y_true[:,i,:]
        pred_point = y_pred[:,i,:]

        rmse, mae, mape = RMSE_MAE_MAPE(real_point, pred_point)

        log_string(log, f"Loss on test data for horizon {i+1}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}")


    rmse, mae, mape = RMSE_MAE_MAPE(y_true, y_pred)
    log_string(log, f"Total test loss - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")


def main(args, device):
    
    dataset_sequence = f"{args.data_dir}/{args.dataset}/{args.dataset}.npz"
    adj_filename = f"{args.data_dir}/{args.dataset}/{args.dataset}.csv"

    adj = get_adjacency_matrix(adj_filename, args.num_nodes)
    adj = torch.from_numpy(adj).unsqueeze(0).to(args.device)

    train_loader, val_loader, test_loader, scaler, threshold_week, threshold_day, sem_adj = load_data(args, adj_filename, dataset_sequence)

    backbone = get_backbones(adj_filename, args.num_nodes, args.input_dim, args.hidden_dim)

    model = SIMBAD(args.device, args.seq_len, backbone, adj, threshold_week, threshold_day, sem_adj, args.num_nodes, args.hidden_dim, args.output_dim, args.scale, args.tau)
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)

    criterion = MaskedMAELoss()
    
    if args.mode == 'train':
        train(args.device, args, model, train_loader, val_loader, test_loader, scaler, optimizer, criterion)
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log_string(log, 'Model successfully loaded')
        test(args.device, args, model, test_loader, scaler, criterion)
    elif args.mode == 'train_continue':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        epoch_continue = checkpoint["epoch"]
        val_loss = checkpoint['loss']
        model.load_state_dict(checkpoint["model_state_dict"])
        log_string(log, 'Model successfully loaded')
        train_continue(args.device, args, model, train_loader, val_loader, test_loader, scaler, optimizer, criterion, epoch_continue, val_loss)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log_string(log, 'Model successfully loaded')
        test(args.device, args, model, test_loader, scaler, criterion)
    elif args.mode == 'test':
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string(log, 'Model successfully loaded')
        test(args.device, args, model, test_loader, scaler, criterion)

    #train(device, args, model, train_loader, val_loader, test_loader, adj, stats, optimizer, criterion)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Model')

    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument("--device", type=str, help="use GPU", default="cuda")
    parser.add_argument("--mode", type=str, help="choose to train or test", default="train")
    parser.add_argument("--log", type=str, help="log", default="logging.log")
    parser.add_argument("--checkpoint", type=str, help="checkpoint model", default='model.pth')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Data prepare
    parser.add_argument("--data_dir", type=str, help="dataset directory", default=config["file"]["directory"])
    parser.add_argument("--dataset", type=str, help="type of traffic data", default=config["file"]["dataset"])

    # Dataset details
    parser.add_argument("--points_per_hour", type=int, help="number of points per hour", default=config["data"]["points_per_hour"])
    parser.add_argument("--num_nodes", type=int, help="number of nodes", default=config["data"]["num_nodes"])
    parser.add_argument("--seq_len", type=int, help="input and output sequence length", default=config["data"]["seq_len"])

    # Model details
    parser.add_argument("--weeks", type=int, help="number of weeks", default=config["model"]["weeks"])
    parser.add_argument("--days", type=int, help="number of days", default=config["model"]["days"])
    parser.add_argument("--hours", type=int, help="number of hours", default=config["model"]["hours"])
    parser.add_argument("--input_dim", type=int, help="number of input dimensions", default=config["model"]["input_dim"])
    parser.add_argument("--hidden_dim", type=int, help="number of hidden dimensions", default=config["model"]["hidden_dim"])
    parser.add_argument("--output_dim", type=int, help="number of output dimensions", default=config["model"]["output_dim"])
    parser.add_argument("--node_ratio", type=float, help="sparsity level of semantic adjacency matrix", default=config["model"]["node_ratio"])
    parser.add_argument("--scale", type=float, help="hardness of mask", default=config["model"]["scale"])
    parser.add_argument("--tau", type=float, help="hardness of softmax", default=config["model"]["tau"])

    # Training details
    parser.add_argument("--train_ratio", type=float, help="training dataset ratio", default=config["train"]["train_ratio"])
    parser.add_argument("--val_ratio", type=float, help="validation dataset ratio", default=config["train"]["val_ratio"])
    parser.add_argument("--batch_size", type=int, help="batch size", default=config["train"]["batch_size"])
    parser.add_argument("--epochs", type=int, help="maximum number of epochs", default=config["train"]["epochs"])
    parser.add_argument("--lr_rate", type=float, help="learning rate", default=config["train"]["lr_rate"])
    parser.add_argument("--early_stop", type=int, help="maximum number of epochs of no improvement in validation", default=config["train"]["early_stop"])
    parser.add_argument("--metric", type=str, help="type of loss for optimization", default=config["train"]["metric"])
    parser.add_argument("--print_every", type=int, help="step of printing training process", default=config["train"]["print_every"])

    args = parser.parse_args()

    device = torch.device(args.device)
    log = open(args.log, 'w')
    log_string(log, f'{args.checkpoint}')
    main(args, device)
