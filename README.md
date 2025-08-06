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

### Dataset Statistics
| Dataset | PEMS03 | PEMS04 | PEMS07 | PEMS08 |
|---------|--------|--------|--------|--------|
| # of nodes | 358 | 307 | 883 | 170 |
| # of timesteps | 26,208 | 16,992 | 28,224 | 17,856 |
| # Granularity | 5 mins | 5 mins | 5 mins | 5 mins |
| # Start time | Sept. 1st 2018 | Jan. 1st 2018 | May 1st 2017 | July 1st 2016 |
| # End time | Nov. 30th 2018 | Feb. 28th 2018 | Aug. 31st 2017 | Aug. 31st 2016 |
| Signals | F | F,S,O | F | F,S,O |

In column titled “Signals”, **F** represents traffic flow, **S** represents traffic speed, and **O** represents traffic occupancy rate.
