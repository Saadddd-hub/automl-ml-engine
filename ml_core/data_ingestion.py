import pandas as pd
import os

def download_data():
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"

    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(url)
    df.to_csv("data/dataset.csv", index=False)

    print("Dataset saved successfully!")