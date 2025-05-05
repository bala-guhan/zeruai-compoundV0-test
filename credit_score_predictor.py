import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import time
from tqdm import tqdm
import os
from datetime import datetime

class CreditScorePredictor:
    def __init__(self, data_dir='data', results_dir='results', credit_scores_dir='credit_scores'):
        """
        Initialize the Credit Score Predictor
        
        Args:
            data_dir (str): Directory containing transaction data
            results_dir (str): Directory to save analysis results
            credit_scores_dir (str): Directory to save credit scores
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.credit_scores_dir = credit_scores_dir
        
        # Create directories if they don't exist
        for directory in [data_dir, results_dir, credit_scores_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize data attributes
        self.timeline_df = None
        self.features_df = None
        self.wallet_scores = None
        self.clustered_df = None
        
        # Set device for GPU acceleration if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_and_preprocess_data(self, data_files):
        """
        Load and preprocess transaction data from JSON files
        
        Args:
            data_files (list): List of paths to transaction data files
        """
        print("Loading and preprocessing data...")
        
        # Initialize empty lists for each transaction type
        deposits = []
        withdraws = []
        borrows = []
        repays = []
        liquidates = []
        
        # Process each file
        for file_path in tqdm(data_files, desc="Processing files"):
            data = pd.read_json(file_path)
            
            # Extract transactions by type
            for tx in data['deposits']:
                if tx:
                    deposits.append(tx)
            for tx in data['withdraws']:
                if tx:
                    withdraws.append(tx)
            for tx in data['borrows']:
                if tx:
                    borrows.append(tx)
            for tx in data['repays']:
                if tx:
                    repays.append(tx)
            for tx in data['liquidates']:
                if tx:
                    liquidates.append(tx)
        
        # Convert to DataFrames
        deposits_df = pd.DataFrame(deposits)
        withdraws_df = pd.DataFrame(withdraws)
        borrows_df = pd.DataFrame(borrows)
        repays_df = pd.DataFrame(repays)
        liquidates_df = pd.DataFrame(liquidates)
        
        # Add action type column
        deposits_df['action_type'] = 'deposits'
        withdraws_df['action_type'] = 'withdraws'
        borrows_df['action_type'] = 'borrows'
        repays_df['action_type'] = 'repays'
        liquidates_df['action_type'] = 'liquidates'
        
        # Combine all transactions
        self.timeline_df = pd.concat([
            deposits_df, withdraws_df, borrows_df, repays_df, liquidates_df
        ], ignore_index=True)
        
        # Convert timestamp to datetime
        self.timeline_df['timestamp'] = pd.to_datetime(self.timeline_df['timestamp'], unit='s')
        
        # Add wallet column
        self.timeline_df['wallet'] = self.timeline_df['account.id']
        
        # Add log-transformed amountUSD
        self.timeline_df['log_amountUSD'] = np.log1p(self.timeline_df['amountUSD'])
        
        print(f"Preprocessed data for {self.timeline_df['wallet'].nunique()} unique wallets")

    def compute_wallet_metrics(self):
        """Compute comprehensive metrics for each wallet"""
        if self.timeline_df is None:
            raise ValueError("No data loaded. Call load_and_preprocess_data first.")
        
        print("Computing wallet metrics...")
        
        # Get unique wallets
        wallets = self.timeline_df['wallet'].unique()
        metrics_list = []
        
        # Process each wallet
        for wallet in tqdm(wallets, desc="Analyzing wallets"):
            wallet_data = self.timeline_df[self.timeline_df['wallet'] == wallet]
            
            # Basic metrics
            metrics = {
                'wallet_address': wallet,
                'first_tx_time': wallet_data['timestamp'].min(),
                'last_tx_time': wallet_data['timestamp'].max(),
                'total_tx_count': len(wallet_data),
                'wallet_age_days': (wallet_data['timestamp'].max() - wallet_data['timestamp'].min()).days,
                'tx_frequency_per_day': len(wallet_data) / ((wallet_data['timestamp'].max() - wallet_data['timestamp'].min()).days + 1)
            }
            
            # Transaction type counts
            for action in ['deposits', 'withdraws', 'borrows', 'repays', 'liquidates']:
                count = len(wallet_data[wallet_data['action_type'] == action])
                metrics[f'{action}_count'] = count
                metrics[f'{action}_proportion'] = count / len(wallet_data) if len(wallet_data) > 0 else 0
            
            # Time-based metrics
            timestamps = wallet_data['timestamp'].values
            if len(timestamps) > 1:
                time_intervals = np.diff(np.sort(timestamps))
                metrics['avg_time_between_txs_hours'] = np.mean(time_intervals) / np.timedelta64(1, 'h')
                metrics['std_time_between_txs_hours'] = np.std(time_intervals) / np.timedelta64(1, 'h')
                metrics['max_time_gap_days'] = np.max(time_intervals) / np.timedelta64(1, 'D')
            else:
                metrics['avg_time_between_txs_hours'] = 0
                metrics['std_time_between_txs_hours'] = 0
                metrics['max_time_gap_days'] = 0
            
            # Asset diversity
            metrics['unique_assets'] = wallet_data['asset.id'].nunique()
            metrics['unique_asset_symbols'] = wallet_data['asset.symbol'].nunique()
            
            # Transaction value metrics
            metrics['avg_tx_log_usd'] = wallet_data['log_amountUSD'].mean()
            metrics['max_tx_log_usd'] = wallet_data['log_amountUSD'].max()
            metrics['min_tx_log_usd'] = wallet_data['log_amountUSD'].min()
            metrics['std_tx_log_usd'] = wallet_data['log_amountUSD'].std()
            metrics['tx_value_variation'] = wallet_data['log_amountUSD'].std() / wallet_data['log_amountUSD'].mean() if wallet_data['log_amountUSD'].mean() > 0 else 0
            
            # Financial metrics
            total_deposited = wallet_data[wallet_data['action_type'] == 'deposits']['log_amountUSD'].sum()
            total_borrowed = wallet_data[wallet_data['action_type'] == 'borrows']['log_amountUSD'].sum()
            total_repaid = wallet_data[wallet_data['action_type'] == 'repays']['log_amountUSD'].sum()
            
            metrics['total_deposited_log_usd'] = total_deposited
            metrics['total_borrowed_log_usd'] = total_borrowed
            metrics['repayment_ratio'] = total_repaid / total_borrowed if total_borrowed > 0 else float('inf')
            metrics['collateral_ratio'] = total_deposited / total_borrowed if total_borrowed > 0 else float('inf')
            
            metrics_list.append(metrics)
        
        self.features_df = pd.DataFrame(metrics_list)
        print(f"Computed {len(self.features_df.columns)} metrics for {len(self.features_df)} wallets")

    def calculate_credit_scores(self):
        """Calculate credit scores for each wallet"""
        if self.features_df is None:
            raise ValueError("No features computed. Call compute_wallet_metrics first.")
        
        print("Calculating credit scores...")
        
        # Define scoring components
        scoring_components = {
            'wallet_longevity': {
                'weight': 0.2,
                'features': ['wallet_age_days', 'tx_frequency_per_day'],
                'normalization': 'minmax'
            },
            'tx_activity': {
                'weight': 0.2,
                'features': ['total_tx_count', 'unique_assets'],
                'normalization': 'minmax'
            },
            'repayment_efficiency': {
                'weight': 0.3,
                'features': ['repayment_ratio', 'collateral_ratio'],
                'normalization': 'minmax'
            },
            'risk_factors': {
                'weight': 0.3,
                'features': ['liquidation_count', 'std_tx_log_usd'],
                'normalization': 'inverse_minmax'
            }
        }
        
        # Calculate component scores
        scores = pd.DataFrame(index=self.features_df.index)
        
        for component, config in scoring_components.items():
            component_scores = []
            for feature in config['features']:
                if config['normalization'] == 'minmax':
                    score = (self.features_df[feature] - self.features_df[feature].min()) / (self.features_df[feature].max() - self.features_df[feature].min())
                else:  # inverse_minmax
                    score = 1 - (self.features_df[feature] - self.features_df[feature].min()) / (self.features_df[feature].max() - self.features_df[feature].min())
                component_scores.append(score)
            
            scores[component] = np.mean(component_scores, axis=0) * config['weight']
        
        # Calculate final score
        self.wallet_scores = pd.DataFrame({
            'wallet_address': self.features_df['wallet_address'],
            'final_score': scores.sum(axis=1) * 100
        })
        
        # Save scores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wallet_scores.to_csv(f'{self.credit_scores_dir}/wallet_scores_{timestamp}.csv', index=False)
        print(f"Results saved to: {self.credit_scores_dir}/wallet_scores_{timestamp}.csv")

    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering on wallet features"""
        if self.features_df is None:
            raise ValueError("No features computed. Call compute_wallet_metrics first.")
        
        print("Performing wallet clustering...")
        
        # Select numerical features
        numerical_features = [
            'wallet_age_days', 'tx_frequency_per_day', 'total_tx_count',
            'unique_assets', 'avg_tx_log_usd', 'std_tx_log_usd',
            'total_deposited_log_usd', 'total_borrowed_log_usd',
            'repayment_ratio', 'collateral_ratio'
        ]
        
        # Preprocess features
        df = self.features_df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in numerical_features:
            df[col] = df[col].fillna(df[col].mean())
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[numerical_features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Calculate cluster centers
        cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=numerical_features
        )
        
        # Calculate cluster statistics
        cluster_stats = df.groupby('cluster')[numerical_features].agg(['mean', 'std', 'count'])
        
        self.clustered_df = df
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'{self.results_dir}/wallet_clusters_{timestamp}.csv', index=False)
        cluster_centers.to_csv(f'{self.results_dir}/cluster_centers_{timestamp}.csv', index=False)
        cluster_stats.to_csv(f'{self.results_dir}/cluster_stats_{timestamp}.csv')
        
        return df, cluster_centers, cluster_stats

    def visualize_results(self):
        """Create visualizations of credit scores and clustering results"""
        if self.wallet_scores is None:
            raise ValueError("No scores calculated. Call calculate_credit_scores first.")
        
        # Set style
        plt.style.use('seaborn')
        
        # Create credit score distribution plot
        plt.figure(figsize=(12, 6))
        plt.hist(self.wallet_scores['final_score'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Wallet Credit Scores', fontsize=14, pad=20)
        plt.xlabel('Credit Score (0-100)', fontsize=12)
        plt.ylabel('Number of Wallets', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add score ranges
        plt.axvline(x=80, color='green', linestyle='--', alpha=0.5, label='Excellent (80-100)')
        plt.axvline(x=60, color='blue', linestyle='--', alpha=0.5, label='Good (60-80)')
        plt.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Fair (40-60)')
        plt.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Poor (20-40)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nCredit Score Distribution:")
        print(f"Total Wallets: {len(self.wallet_scores)}")
        print(f"Mean Score: {self.wallet_scores['final_score'].mean():.2f}")
        print(f"Median Score: {self.wallet_scores['final_score'].median():.2f}")
        print(f"Standard Deviation: {self.wallet_scores['final_score'].std():.2f}")
        
        if self.clustered_df is not None:
            # Create cluster visualization
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Wallet Age vs Transaction Frequency
            plt.subplot(2, 2, 1)
            sns.scatterplot(data=self.clustered_df, x='wallet_age_days', y='tx_frequency_per_day', 
                          hue='cluster', palette='viridis')
            plt.title('Wallet Age vs Transaction Frequency')
            
            # Plot 2: Total Transactions vs Unique Assets
            plt.subplot(2, 2, 2)
            sns.scatterplot(data=self.clustered_df, x='total_tx_count', y='unique_assets', 
                          hue='cluster', palette='viridis')
            plt.title('Total Transactions vs Unique Assets')
            
            # Plot 3: Average Transaction Value vs Repayment Ratio
            plt.subplot(2, 2, 3)
            sns.scatterplot(data=self.clustered_df, x='avg_tx_log_usd', y='repayment_ratio', 
                          hue='cluster', palette='viridis')
            plt.title('Average Transaction Value vs Repayment Ratio')
            
            # Plot 4: Collateral Ratio vs Borrowed Amount
            plt.subplot(2, 2, 4)
            sns.scatterplot(data=self.clustered_df, x='collateral_ratio', y='total_borrowed_log_usd', 
                          hue='cluster', palette='viridis')
            plt.title('Collateral Ratio vs Total Borrowed')
            
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    predictor = CreditScorePredictor()
    predictor.load_and_preprocess_data(['raw_datasets/compoundV2_transactions_ethereum_chunk_0.json'])
    predictor.compute_wallet_metrics()
    predictor.calculate_credit_scores()
    predictor.perform_clustering()
    predictor.visualize_results() 