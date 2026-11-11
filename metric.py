import numpy as np
import torch

def _get_mask(y_true, null_val):
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    return np.nan_to_num(mask)


def RMSE(y_true, y_pred, null_val=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = _get_mask(y_true, null_val)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred, null_val=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = _get_mask(y_true, null_val)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = _get_mask(y_true, null_val)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def RMSE_MAE_MAPE(y_true, y_pred, null_val=0.0):
    rmse = RMSE(y_true, y_pred, null_val)
    mae = MAE(y_true, y_pred, null_val)
    mape = MAPE(y_true, y_pred, null_val)
    return (
        rmse,
        mae,
        mape
    )

def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    """Compute mean absolute error while masking missing targets."""

    def __init__(self, null_val=0.0):
        self.null_val = null_val

    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels):
        return masked_mae_loss(preds, labels, self.null_val)
