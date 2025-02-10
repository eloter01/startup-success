# src/data/load_data.py
import kagglehub
import pandas as pd

def download_dataset():
    path = kagglehub.dataset_download("yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase")
    return path

def load_raw_data(path):
    files = os.listdir(path)
    first_file = files[0]
    file_path = os.path.join(path, first_file)
    
    if first_file.endswith('.csv'):
        df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    path = download_dataset()
    df = load_raw_data(path)
    df.to_csv('data/raw/startups.csv', index=False)