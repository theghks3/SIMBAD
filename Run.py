import argparse
import ast
import configparser
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.optim as optim

from data_preparation import get_final_dataset, read_and_generate_dataset
from metric import MaskedMAELoss, RMSE_MAE_MAPE
from SIMBAD import SIMBAD
from uncertainty import compute_uncertainty_loss
from utils import get_adjacency_matrix, log_string


def log_dataset_shapes(dataset):
    for split_name, display_name in (
        ('train', 'Train'),
        ('val', 'Validation'),
        ('test', 'Test'),
    ):
        shapes = ', '.join(
            f'{key}: {dataset[split_name][key].shape}'
            for key in ('week', 'day', 'hour')
        )
        log_string(log, f'{display_name} {shapes}')


def load_data(args, dataset_sequence):
    all_data = read_and_generate_dataset(dataset_sequence, args.weeks, args.days, args.hours, args.seq_len, args.input_dim,
                                        args.train_ratio, args.val_ratio, args.points_per_hour,
                                        args.similarity_window,
                                        args.threshold_multiplier,
                                        args.data_key,
                                        args.target_channel)
    log_dataset_shapes(all_data)

    actual_nodes = all_data['train']['week'].shape[1]
    if actual_nodes != args.num_nodes:
        raise ValueError(
            f"num_nodes={args.num_nodes}, but dataset contains {actual_nodes} nodes"
        )

    train_loader, val_loader, test_loader, stats = get_final_dataset(all_data, args.batch_size)

    return train_loader, val_loader, test_loader, stats, all_data['threshold']['sim']


def validate_args(args):
    if args.points_per_hour <= 0:
        raise ValueError("points_per_hour must be positive")
    if args.seq_len <= 0 or args.input_dim <= 0 or args.num_nodes <= 0:
        raise ValueError("seq_len, input_dim, and num_nodes must be positive")
    if args.weeks < 0 or args.days < 0 or args.weeks + args.days < 1:
        raise ValueError("weeks and days must be non-negative with a positive sum")
    if args.hours <= 0:
        raise ValueError("hours must be positive")
    if not 0 < args.similarity_window <= args.seq_len:
        raise ValueError("similarity_window must be between 1 and seq_len")
    if args.hidden_dim <= 0 or args.num_layers <= 0 or args.num_heads <= 0:
        raise ValueError("hidden_dim, num_layers, and num_heads must be positive")
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if args.tau <= 0:
        raise ValueError("tau must be positive")
    if args.lambda_u < 0 or args.threshold_multiplier < 0:
        raise ValueError("lambda_u and threshold_multiplier must be non-negative")
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    if args.batch_size <= 0 or args.epochs <= 0 or args.print_every <= 0:
        raise ValueError("batch_size, epochs, and print_every must be positive")
    if args.lr_rate <= 0 or args.early_stop <= 0:
        raise ValueError("lr_rate and early_stop must be positive")
    if any(milestone <= 0 for milestone in args.patience):
        raise ValueError("scheduler milestones must be positive")
    if not 0 < args.decay <= 1:
        raise ValueError("decay must be in (0, 1]")

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

        for week, day, hour, sim, target in val_loader:
            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            sim = sim.to(device)
            pred, _ = model([week, day, hour, sim])

            y_true_list.append(target.to(device).detach().cpu())
            y_pred_list.append(pred.detach().cpu())

        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        y_pred = scaler.inverse_transform(y_pred)

        loss = loss_func(y_pred, y_true)
    log_string(log, f'Epoch: {epoch}, validation loss: {loss:.2f}')

    return loss

def train(device, args, model, train_loader, val_loader, scaler, optimizer, loss_func):
    global_step = 0
    best_epoch = 0
    best_val_loss = float('inf')
    early_stop = 0
    start_time = time()

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.patience,
        gamma=args.decay
    )

    for epoch in range(args.epochs):
        log_string(log, f'Epoch: {epoch+1}')

        model.train()

        for week, day, hour, sim, sim_target, target in train_loader:
            optimizer.zero_grad()

            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            sim = sim.to(device)
            sim_target = sim_target.to(device)

            outputs, loss_compute = model([week, day, hour, sim])
            unc_loss = compute_uncertainty_loss(sim_target, loss_compute[0], loss_compute[1])
            outputs = scaler.inverse_transform(outputs)
            pred_loss = loss_func(outputs, target.to(device))
            loss = pred_loss + args.lambda_u * unc_loss
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1

            if global_step % args.print_every == 0:
                log_string(log, f'Global step: {global_step}, Training loss: {training_loss:.2f}, Time: {time() - start_time:.2f}s')

        val_loss = compute_val_loss(device, model, val_loader, scaler, loss_func, epoch+1)
        scheduler.step()
        log_string(log, f"Optimizer learning rate: {optimizer.param_groups[0]['lr']}")
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

def test(device, args, model, test_loader, scaler):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for week, day, hour, sim, target in test_loader:
            week = week.to(device)
            day = day.to(device)
            hour = hour.to(device)
            sim = sim.to(device)
            output, _ = model([week, day, hour, sim])
            test_t = target.to(device).permute(0,2,1).unsqueeze(-1)
            output = output.permute(0,2,1).unsqueeze(-1)
            y_true.append(test_t)
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()[...,0]
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()[...,0]
    y_pred = scaler.inverse_transform(y_pred)

    for i in range(y_true.shape[1]):
        real_point = y_true[:,i,:]
        pred_point = y_pred[:,i,:]

        rmse, mae, mape = RMSE_MAE_MAPE(real_point, pred_point, args.null_val)

        log_string(log, f"Loss on test data for horizon {i+1}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}")


    rmse, mae, mape = RMSE_MAE_MAPE(y_true, y_pred, args.null_val)
    log_string(log, f"Total test loss - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")


def main(args, device):

    dataset_dir = Path(args.data_dir) / args.dataset
    dataset_sequence = dataset_dir / args.signal_file
    adj_filename = dataset_dir / args.adjacency_file
    adj = get_adjacency_matrix(adj_filename, args.num_nodes)
    adj = torch.from_numpy(adj).unsqueeze(0).to(device)

    train_loader, val_loader, test_loader, scaler, threshold = load_data(args, dataset_sequence)

    model = SIMBAD(
        device=device,
        num_for_prediction=args.seq_len,
        adj=adj,
        threshold=threshold,
        stat=scaler,
        num_of_vertices=args.num_nodes,
        tau=args.tau,
        dropout=args.dropout,
        similarity_window=args.similarity_window,
        num_weeks=args.weeks,
        num_days=args.days,
        num_hours=args.hours,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)

    criterion = MaskedMAELoss(args.null_val)

    if args.mode == 'train':
        train(device, args, model, train_loader, val_loader, scaler['target'], optimizer, criterion)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log_string(log, 'Model successfully loaded')
        test(device, args, model, test_loader, scaler['target'])
    elif args.mode == 'test':
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string(log, 'Model successfully loaded')
        test(device, args, model, test_loader, scaler['target'])

if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', required=True)
    config_args, _ = config_parser.parse_known_args()

    config = configparser.ConfigParser()
    if not config.read(config_args.config):
        config_parser.error(f'configuration file not found: {config_args.config}')

    parser = argparse.ArgumentParser(description='Pytorch Model')

    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--device", type=str, help="use GPU", default="cuda")
    parser.add_argument("--mode", choices=("train", "test"), help="execution mode", default="train")
    parser.add_argument("--log", type=str, help="log", default="logging.log")
    parser.add_argument("--checkpoint", type=str, help="checkpoint model", default='model.pth')
    
    # Data prepare
    parser.add_argument("--data_dir", type=str, help="dataset directory", default=config["file"]["directory"])
    parser.add_argument("--dataset", type=str, help="type of traffic data", default=config["file"]["dataset"])
    parser.add_argument("--data_key", type=str, help="array key in the NPZ file", default=config["file"]["data_key"])
    parser.add_argument("--signal_file", type=str, help="signal NPZ filename", default=config["file"]["signal_file"])
    parser.add_argument("--adjacency_file", type=str, help="adjacency CSV filename", default=config["file"]["adjacency_file"])

    # Dataset details
    parser.add_argument("--points_per_hour", type=int, help="number of points per hour", default=config["data"]["points_per_hour"])
    parser.add_argument("--num_nodes", type=int, help="number of nodes", default=config["data"]["num_nodes"])
    parser.add_argument("--seq_len", type=int, help="input and output sequence length", default=config["data"]["seq_len"])
    parser.add_argument("--target_channel", type=int, help="feature index to predict", default=config["data"]["target_channel"])
    parser.add_argument("--null_val", type=float, help="missing target value", default=config["data"]["null_val"])

    # Model details
    parser.add_argument("--weeks", type=int, help="number of weeks", default=config["model"]["weeks"])
    parser.add_argument("--days", type=int, help="number of days", default=config["model"]["days"])
    parser.add_argument("--hours", type=int, help="number of hours", default=config["model"]["hours"])
    parser.add_argument("--input_dim", type=int, help="number of input dimensions", default=config["model"]["input_dim"])
    parser.add_argument("--hidden_dim", type=int, help="hidden feature dimension", default=config["model"]["hidden_dim"])
    parser.add_argument("--num_layers", type=int, help="number of layers per STM stack", default=config["model"]["num_layers"])
    parser.add_argument("--num_heads", type=int, help="number of attention heads", default=config["model"]["num_heads"])
    parser.add_argument("--dropout", type=float, help="dropout level", default=config["model"]["dropout"])
    parser.add_argument("--similarity_window", type=int, help="window length for L1 similarity", default=config["model"]["similarity_window"])
    parser.add_argument("--threshold_multiplier", type=float, help="IQR multiplier for similarity threshold", default=config["model"]["threshold_multiplier"])
    parser.add_argument("--lambda_u", type=float, default=config["model"]["lambda_u"])
    parser.add_argument("--tau", type=float, default=config["model"]["tau"])

    # Training details
    parser.add_argument("--train_ratio", type=float, help="training dataset ratio", default=config["train"]["train_ratio"])
    parser.add_argument("--val_ratio", type=float, help="validation dataset ratio", default=config["train"]["val_ratio"])
    parser.add_argument("--batch_size", type=int, help="batch size", default=config["train"]["batch_size"])
    parser.add_argument("--epochs", type=int, help="maximum number of epochs", default=config["train"]["epochs"])
    parser.add_argument("--lr_rate", type=float, help="learning rate", default=config["train"]["lr_rate"])
    parser.add_argument("--early_stop", type=int, help="maximum number of epochs of no improvement in validation", default=config["train"]["early_stop"])
    parser.add_argument("--patience", nargs="+", type=int, help="learning weight decay epoch", default=ast.literal_eval(config["train"]["patience"]))
    parser.add_argument("--decay", type=float, help="learning-rate decay factor", default=config["train"]["decay"])
    parser.add_argument("--print_every", type=int, help="step of printing training process", default=config["train"]["print_every"])
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    device = torch.device(args.device)
    with open(args.log, 'w') as log:
        log_string(log, str(args))
        main(args, device)
