# DeFi Wallet Credit Scoring System

A comprehensive system for analyzing and scoring DeFi wallet behavior using transaction data.

## Features

- Transaction data preprocessing and analysis
- Comprehensive wallet metrics computation
- Credit score calculation based on multiple factors
- K-means clustering for wallet segmentation
- Interactive visualizations
- GPU-accelerated computations (when available)

## Project Structure

```
.
├── credit_score_predictor.py    # Main class for credit scoring and analysis
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── raw_datasets/                # Raw transaction data
├── credit_scores/               # Generated credit scores
└── results/                     # Analysis results and visualizations
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/defi-credit-scoring.git
cd defi-credit-scoring
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from credit_score_predictor import CreditScorePredictor

# Initialize the predictor
predictor = CreditScorePredictor()

# Load and preprocess data
predictor.load_and_preprocess_data(['raw_datasets/your_transactions.json'])

# Compute wallet metrics
predictor.compute_wallet_metrics()

# Calculate credit scores
predictor.calculate_credit_scores()

# Perform clustering
predictor.perform_clustering(n_clusters=5)

# Visualize results
predictor.visualize_results()
```

### Advanced Usage

```python
# Initialize with custom directories
predictor = CreditScorePredictor(
    data_dir='custom_data',
    results_dir='custom_results',
    credit_scores_dir='custom_scores'
)

# Load multiple data files
predictor.load_and_preprocess_data([
    'raw_datasets/file1.json',
    'raw_datasets/file2.json'
])

# Perform clustering with different number of clusters
predictor.perform_clustering(n_clusters=7)
```

## Features Calculated

The system computes the following metrics for each wallet:

1. **Basic Metrics**

   - Wallet age
   - Transaction frequency
   - Total transaction count
   - First and last transaction times

2. **Transaction Analysis**

   - Transaction type counts (deposits, withdrawals, borrows, repays, liquidates)
   - Time between transactions
   - Asset diversity

3. **Financial Metrics**

   - Average transaction value
   - Total deposited amount
   - Total borrowed amount
   - Repayment ratio
   - Collateral ratio

4. **Credit Score Components**
   - Wallet longevity (20%)
   - Transaction activity (20%)
   - Repayment efficiency (30%)
   - Risk factors (30%)

## Results

The system generates the following outputs:

1. **Credit Scores**

   - Individual wallet scores (0-100)
   - Score distribution statistics
   - Score range classifications (Excellent, Good, Fair, Poor)

2. **Clustering Results**

   - Wallet clusters
   - Cluster centers
   - Cluster statistics
   - Visualizations of cluster characteristics

3. **Visualizations**
   - Credit score distribution
   - Cluster scatter plots
   - Feature relationships

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python and popular data science libraries
- Inspired by traditional credit scoring systems
- Adapted for DeFi wallet analysis
