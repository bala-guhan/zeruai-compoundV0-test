import os
import shutil
from pathlib import Path

def cleanup_directory():
    """Clean up and organize the project directory structure"""
    
    # Create necessary directories if they don't exist
    directories = ['src', 'notebooks', 'data', 'results', 'credit_scores']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Move files to appropriate directories
    files_to_move = {
        'credit_score_predictor.py': 'src/',
        'credit_score_visualization.py': 'src/',
        'wallet_metrics_updated.py': 'src/',
        'defi.ipynb': 'notebooks/',
        'raw_datasets/': 'data/'
    }
    
    # Move files
    for file, dest in files_to_move.items():
        if os.path.exists(file):
            if os.path.isdir(file):
                # Move directory contents
                for item in os.listdir(file):
                    src_path = os.path.join(file, item)
                    dest_path = os.path.join(dest, item)
                    if os.path.exists(dest_path):
                        if os.path.isfile(dest_path):
                            os.remove(dest_path)
                        else:
                            shutil.rmtree(dest_path)
                    shutil.move(src_path, dest_path)
                # Remove empty directory
                os.rmdir(file)
            else:
                # Move file
                dest_path = os.path.join(dest, os.path.basename(file))
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(file, dest_path)
    
    # Remove unnecessary files
    files_to_remove = ['ps.txt']
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
    
    print("Directory cleanup completed!")
    print("\nNew directory structure:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    cleanup_directory() 