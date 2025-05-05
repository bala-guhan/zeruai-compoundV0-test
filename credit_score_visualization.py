import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_credit_scores(score_file_path):
    """
    Create a comprehensive visualization of credit scores with proper labels and statistics
    
    Args:
        score_file_path (str): Path to the credit scores CSV file
    """
    # Load the credit scores data
    comp2 = pd.read_csv(score_file_path)
    
    # Set the style
    plt.style.use('seaborn')
    
    # Create a figure with proper styling
    plt.figure(figsize=(12, 6))
    
    # Plot the histogram
    plt.hist(comp2['final_score'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    
    # Add labels and title
    plt.title('Distribution of Wallet Credit Scores', fontsize=14, pad=20)
    plt.xlabel('Credit Score (0-100)', fontsize=12)
    plt.ylabel('Number of Wallets', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate statistics
    mean_score = comp2['final_score'].mean()
    median_score = comp2['final_score'].median()
    std_score = comp2['final_score'].std()
    
    # Add descriptive statistics as text
    stats_text = f'Mean: {mean_score:.2f}\nMedian: {median_score:.2f}\nStd Dev: {std_score:.2f}'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8))
    
    # Add score ranges
    plt.axvline(x=80, color='green', linestyle='--', alpha=0.5, label='Excellent (80-100)')
    plt.axvline(x=60, color='blue', linestyle='--', alpha=0.5, label='Good (60-80)')
    plt.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Fair (40-60)')
    plt.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Poor (20-40)')
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print score distribution statistics
    print("\nCredit Score Distribution:")
    print(f"Total Wallets: {len(comp2)}")
    print(f"Mean Score: {mean_score:.2f}")
    print(f"Median Score: {median_score:.2f}")
    print(f"Standard Deviation: {std_score:.2f}")
    print("\nScore Ranges:")
    print(f"Excellent (80-100): {len(comp2[comp2['final_score'] >= 80])} wallets")
    print(f"Good (60-80): {len(comp2[(comp2['final_score'] >= 60) & (comp2['final_score'] < 80)])} wallets")
    print(f"Fair (40-60): {len(comp2[(comp2['final_score'] >= 40) & (comp2['final_score'] < 60)])} wallets")
    print(f"Poor (20-40): {len(comp2[(comp2['final_score'] >= 20) & (comp2['final_score'] < 40)])} wallets")
    print(f"Very Poor (0-20): {len(comp2[comp2['final_score'] < 20])} wallets")
    
    # Create a box plot for additional insights
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=comp2['final_score'])
    plt.title('Credit Score Distribution Box Plot', fontsize=14)
    plt.xlabel('Credit Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage:
# from credit_score_visualization import visualize_credit_scores
# visualize_credit_scores('credit_scores/wallet_scores_20250505_210739.csv') 