import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def compute_wallet_metrics(timeline_df):
    """
    Calculate comprehensive metrics for each wallet from the processed transaction data
    
    Args:
        timeline_df (pd.DataFrame): DataFrame containing processed transaction data with columns:
            - wallet: wallet address
            - timestamp: transaction timestamp
            - amountUSD: transaction amount in USD
            - action_type: type of transaction (deposits, withdraws, borrows, repays, liquidates)
            - asset.id: asset identifier
            - asset.symbol: asset symbol
            - log_amountUSD: log-transformed amount in USD
        
    Returns:
        pd.DataFrame: DataFrame containing computed metrics for each wallet
    """
    if timeline_df is None:
        raise ValueError("Transaction data not found. Please provide valid transaction data.")
        
    print("Computing wallet metrics...")
    start_time = time.time()
    
    # Get unique wallet addresses
    wallet_addresses = timeline_df['wallet'].unique()
    
    # Initialize storage for wallet metrics
    wallet_metrics = {}
    
    # Process each wallet with progress tracking
    wallet_progress = tqdm(wallet_addresses, desc="Analyzing wallets", unit="wallet")
    
    for wallet in wallet_progress:
        # Update progress description periodically
        if wallet_progress.n % 100 == 0:
            wallet_progress.set_description(f"Analyzing wallet {wallet_progress.n}/{len(wallet_addresses)}")
        
        # Filter transactions for current wallet
        wallet_transactions = timeline_df[timeline_df['wallet'] == wallet]
        
        # Basic transaction metrics
        metrics = {
            'wallet_address': wallet,
            'first_tx_time': wallet_transactions['timestamp'].min(),
            'last_tx_time': wallet_transactions['timestamp'].max(),
            'total_tx_count': len(wallet_transactions),
        }
        
        # Calculate wallet age
        metrics['wallet_age_days'] = (metrics['last_tx_time'] - metrics['first_tx_time']).total_seconds() / (60*60*24)
        
        # Transaction frequency
        if metrics['wallet_age_days'] > 0:
            metrics['tx_frequency_per_day'] = metrics['total_tx_count'] / metrics['wallet_age_days']
        else:
            metrics['tx_frequency_per_day'] = metrics['total_tx_count']
        
        # Transaction type analysis
        tx_type_counts = wallet_transactions['action_type'].value_counts()
        total_txs = len(wallet_transactions)
        
        for tx_type in ['deposits', 'withdraws', 'borrows', 'repays', 'liquidates']:
            count = tx_type_counts.get(tx_type, 0)
            metrics[f'{tx_type}_count'] = count
            metrics[f'{tx_type}_proportion'] = count / total_txs if total_txs > 0 else 0
        
        # Liquidation history
        metrics['has_been_liquidated'] = 1 if metrics['liquidates_count'] > 0 else 0
        
        # Deposit and withdrawal analysis
        deposits = wallet_transactions[wallet_transactions['action_type'] == 'deposits']
        withdraws = wallet_transactions[wallet_transactions['action_type'] == 'withdraws']
        
        # Financial metrics using log_amountUSD
        metrics['total_deposited_log_usd'] = deposits['log_amountUSD'].sum() if not deposits.empty else 0
        metrics['total_withdrawn_log_usd'] = withdraws['log_amountUSD'].sum() if not withdraws.empty else 0
        metrics['net_position_log_usd'] = metrics['total_deposited_log_usd'] - metrics['total_withdrawn_log_usd']
        
        # Borrowing analysis
        borrows = wallet_transactions[wallet_transactions['action_type'] == 'borrows']
        repays = wallet_transactions[wallet_transactions['action_type'] == 'repays']
        
        metrics['total_borrowed_log_usd'] = borrows['log_amountUSD'].sum() if not borrows.empty else 0
        metrics['total_repaid_log_usd'] = repays['log_amountUSD'].sum() if not repays.empty else 0
        
        # Repayment analysis
        if metrics['total_borrowed_log_usd'] > 0:
            metrics['repayment_ratio'] = metrics['total_repaid_log_usd'] / metrics['total_borrowed_log_usd']
        else:
            metrics['repayment_ratio'] = 1.0
        
        # Transaction timing analysis
        if len(wallet_transactions) > 1:
            tx_times = wallet_transactions['timestamp'].sort_values()
            time_intervals = tx_times.diff().dropna()
            metrics['avg_time_between_txs_hours'] = time_intervals.mean().total_seconds() / 3600
            metrics['std_time_between_txs_hours'] = time_intervals.std().total_seconds() / 3600 if len(time_intervals) > 1 else 0
            metrics['max_time_gap_days'] = time_intervals.max().total_seconds() / (24*3600)
        else:
            metrics['avg_time_between_txs_hours'] = 0
            metrics['std_time_between_txs_hours'] = 0
            metrics['max_time_gap_days'] = 0
        
        # Asset diversity
        metrics['unique_assets'] = wallet_transactions['asset.id'].nunique()
        metrics['unique_asset_symbols'] = wallet_transactions['asset.symbol'].nunique()
        
        # Transaction value analysis using log_amountUSD
        if 'log_amountUSD' in wallet_transactions.columns:
            metrics['avg_tx_log_usd'] = wallet_transactions['log_amountUSD'].mean()
            metrics['max_tx_log_usd'] = wallet_transactions['log_amountUSD'].max()
            metrics['min_tx_log_usd'] = wallet_transactions['log_amountUSD'].min()
            metrics['std_tx_log_usd'] = wallet_transactions['log_amountUSD'].std() if len(wallet_transactions) > 1 else 0
            metrics['tx_value_variation'] = metrics['std_tx_log_usd'] / metrics['avg_tx_log_usd'] if metrics['avg_tx_log_usd'] > 0 else 0
        
        # Collateral analysis
        if metrics['total_borrowed_log_usd'] > 0:
            metrics['collateral_ratio'] = metrics['total_deposited_log_usd'] / metrics['total_borrowed_log_usd']
        else:
            metrics['collateral_ratio'] = float('inf')
        
        # Store metrics for this wallet
        wallet_metrics[wallet] = metrics
        
        # Update progress stats periodically
        if wallet_progress.n % 500 == 0:
            wallet_progress.set_postfix(elapsed=f"{time.time() - start_time:.1f}s")
    
    # Convert to DataFrame
    wallet_metrics_df = pd.DataFrame(list(wallet_metrics.values()))
    elapsed = time.time() - start_time
    print(f"Computed {len(wallet_metrics_df.columns)} metrics for {len(wallet_metrics_df)} wallets in {elapsed:.2f} seconds")
    
    return wallet_metrics_df