import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from scipy.special import gamma as gamma_fn


class RNNGARCH(nn.Module):

    def __init__(self, hidden_size):
        super(RNNGARCH, self).__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(2, hidden_size) 
        self.W_hh = nn.Linear(hidden_size, hidden_size)  
        self.fc = nn.Linear(hidden_size, 1)  


    def forward(self, residuals, return_hidden=False):
        """
        Forward pass through RNN-GARCH.

        Args:
            residuals (Tensor): (batch_size, sequence_length)
            return_hidden (bool): If True, also return the final hidden state

        Returns:
            cond_vars (Tensor): (batch_size, sequence_length)
            h_t (Tensor, optional): (batch_size, hidden_size)
        """
        batch_size, seq_len = residuals.shape
        
        sigma_sq = torch.var(residuals, dim=1, keepdim=True)  # (batch_size, 1)
        cond_vars = [sigma_sq]
        h_t = torch.zeros(batch_size, self.hidden_size, device=residuals.device)

        for t in range(1, seq_len):
            epsilon_sq = residuals[:, t - 1]**2
            input_t = torch.stack([epsilon_sq, sigma_sq.squeeze(1)], dim=1)  # (batch_size, 2)

            h_t = self.W_ih(input_t) + self.W_hh(h_t) # (batch_size, hidden_size)
            sigma_sq = F.softplus(self.fc(h_t)) + 1e-6
            cond_vars.append(sigma_sq)

        cond_vars = torch.cat(cond_vars, dim=1)  # (batch_size, seq_len)

        if return_hidden:
            return cond_vars, h_t
        return cond_vars


def create_windows(residuals, window_size):
    """
    Create overlapping sliding windows from a residual series.

    Args:
        residuals (Tensor): (T_train,)
        window_size (int): Size of each window.

    Returns:
        windows (Tensor): (num_windows, window_size)
    """
    T = residuals.shape[0]
    num_windows = T - window_size + 1
    
    windows = []
    for i in range(num_windows):
        window = residuals[i:i + window_size]
        windows.append(window)
    
    windows = torch.stack(windows, dim=0)  # (num_windows, window_size)
    return windows


def negative_llk(residuals, cond_vars, dist='normal'):
    """
    Compute the negative average log-likelihood under specified distribution.

    Args:
        residuals (Tensor): (batch_size, sequence_length)
        cond_vars (Tensor): (batch_size, sequence_length)
        dist (str): Distribution type: 'normal' or 'ged'

    Returns:
        loss (Tensor): scalar
    """
    if dist == 'normal':
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=cond_vars.device))
        ll = -0.5 * (log_2pi + torch.log(cond_vars) + residuals**2 / cond_vars)
    elif dist in ['laplace', 'ged']:
        beta_ged = 1 if dist == 'laplace' else 1.5
        gamma_1 = gamma_fn(1 / beta_ged)
        gamma_3 = gamma_fn(3 / beta_ged)
        alpha_ged = torch.sqrt(cond_vars * gamma_1 / gamma_3)
        norm_const = beta_ged / (2 * alpha_ged * gamma_1)
        ll = torch.log(norm_const) - (torch.abs(residuals) / alpha_ged) ** beta_ged
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    return -ll.sum() / residuals.size(0) 


def evaluate_forecast(predicted, realized):
    """
    Compute evaluation metrics for variance forecasts.

    Args:
        predicted (array-like): Forecasted variances
        realized (array-like): Realized variances

    Returns:
        metrics (dict): Dictionary with MSE, MAE, and HMSE
    """
    predicted = np.array(predicted)
    realized = np.array(realized)

    mse = np.mean((predicted - realized)**2)
    mae = np.mean(np.abs(predicted - realized))
    hmse = np.mean((1 - predicted / realized)**2)

    return {
        'MSE': mse,
        'MAE': mae,
        'HMSE': hmse
    }


def main():
    # === Set seed ===
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # === Adjustable parameters ===
    asset = 'BTC'
    window_size = 1000
    hidden_size = 8
    num_epochs = 100
    batch_size = 32
    lr = 1e-2
    dist = 'ged'
    train = False

    # === Load residuals and realized variances ===
    residuals_df = pd.read_csv('../data/daily_residuals.csv', index_col=0, parse_dates=True)
    realized_vars_df = pd.read_csv('../data/daily_realized_vars.csv', index_col=0, parse_dates=True)
    cvx_df = pd.read_csv('../data/daily_cvx.csv', index_col=0, parse_dates=True)

    common_idx = realized_vars_df.index.intersection(cvx_df.index)
    residuals = residuals_df[asset].loc[common_idx].to_numpy()
    realized_vars = realized_vars_df[asset].loc[common_idx].to_numpy()

    residuals = torch.tensor(residuals, dtype=torch.float32)
    realized_vars = torch.tensor(realized_vars, dtype=torch.float32) 

    # === Split data into train/test set ===
    split_idx = int(0.8 * len(residuals))

    residuals_train = residuals[:split_idx]
    residuals_test = residuals[split_idx:]
    realized_vars_train = realized_vars[:split_idx]
    realized_vars_test = realized_vars[split_idx:]

    # === Create training windows ===
    train_windows = create_windows(residuals_train, window_size)
    train_loader = DataLoader(train_windows, batch_size=batch_size, shuffle=True)
    
    # === Initialize model ===
    model = RNNGARCH(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # === Training loop ===
    if train:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for residuals_batch in train_loader:
                cond_vars_batch = model(residuals_batch)
                loss = negative_llk(residuals_batch, cond_vars_batch, dist=dist)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}')

            scheduler.step()

        torch.save(model.state_dict(), f'../states/rnn_garch_{dist}.pth')
    else:
        model.load_state_dict(torch.load(f'../states/rnn_garch_{dist}.pth'))

    # === In-sample fit ===
    model.eval()

    with torch.no_grad():
        residuals_train = residuals_train.unsqueeze(0)
        cond_vars_train = model(residuals_train)
        neg_llk = negative_llk(residuals_train, cond_vars_train, dist=dist)

    llk_train = -neg_llk.item()
    print('In-Sample Model Fit:')
    print(f'Log-Likelihood = {llk_train:.2f}')

    cond_vars_train = cond_vars_train.squeeze(0).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(cond_vars_train, label='Fitted', linewidth=1)
    plt.plot(realized_vars_train.numpy(), label='Realized', linewidth=1)
    plt.title(f'RNN-GARCH(1,1) In-Sample Analysis | Distribution = {dist}')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.ylim(-10, 250)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # === Out-of-sample performance ===
    forecasted_vars = np.zeros(len(residuals_test))

    for t in range(len(residuals_test)):
        residuals_window = residuals[split_idx + t - window_size:split_idx + t].unsqueeze(0)
        cond_vars_window, h_t = model(residuals_window, return_hidden=True)

        last_residual = residuals_window[0, -1].item()
        last_var = cond_vars_window[0, -1].item()
        last_input = torch.tensor([[last_residual**2, last_var]], dtype=torch.float32)

        with torch.no_grad():
            h_t = model.W_ih(last_input) + model.W_hh(h_t)
            pred_var = F.softplus(model.fc(h_t)) + 1e-6

        forecasted_vars[t] = pred_var.item()

    metrics = evaluate_forecast(forecasted_vars, realized_vars_test[:len(forecasted_vars)])
    print('\nOut-of-Sample Performance:')
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"HMSE: {metrics['HMSE']:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(forecasted_vars, label='Predicted (1-Day)', linewidth=1)
    plt.plot(realized_vars_test.numpy(), label='Realized', linewidth=1)
    plt.title(f'RNN-GARCH(1,1) Out-of-Sample Analysis | Distribution = {dist}')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.ylim(0, 35)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
