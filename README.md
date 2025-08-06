# Aperiodicity-Robust Spatio-Temporal Traffic Forecasting (SIMBAD)

## Required Packages
```
pytorch == 2.7.1
numpy == 2.2.5
torch-geometric == 2.6.1
```

## Training and Testing Commands
Train SIMBAD through the following commands for the 4 PEMS datasets.

For argument "mode", train is default.
```
python Run.py --data_type PEMS03 --num_vertices 358 --checkpoint PEMS03.pth --mode train
python Run.py --data_type PEMS04 --num_vertices 307 --checkpoint PEMS04.pth --mode train
python Run.py --data_type PEMS07 --num_vertices 883 --checkpoint PEMS07.pth --mode train --batch_size 32
python Run.py --data_type PEMS08 --num_vertices 170 --checkpoint PEMS08.pth --mode train
```

Test SIMBAD with a saved model (based on best validation loss) through the following command.

```
python Run.py --data_type PEMS08 --num_vertices 170 --checkpoint PEMS08.pth --mode test
```

Continue training if terminated although training is not finished through the following command.

```
python Run.py --data_type PEMS08 --num_vertices 170 --checkpoint PEMS08.pth --mode train_continue
```

## Datasets
We used four traffic datasets PEMS03, PEMS04, PEMS07 and PEMS08 which are collected by California Transportation Agencies (CalTrans) Performance Measurement System (PeMS) in real time every 30 seconds.

The collected data is aggregated to 5 minutes, meaning there are 12 points for each hour.

Datasets can be downloaded from [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data).
