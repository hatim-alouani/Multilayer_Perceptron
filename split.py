import sys
import pandas as pd

# split data
try:
    df = pd.read_csv('data.csv', header=None).sample(frac=1, random_state=0)
except FileNotFoundError:
    print("Error: 'data.csv' not found.")
    sys.exit(1)

cut = int(len(df) * 0.8)
df.iloc[:cut].to_csv('data_train.csv', header=False, index=False)
df.iloc[cut:].to_csv('data_valid.csv', header=False, index=False)
