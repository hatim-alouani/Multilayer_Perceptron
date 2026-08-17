import sys
import pandas as pd

def main():
    df = pd.read_csv('data.csv', header=None).sample(frac=1, random_state=0)
    cut = int(len(df) * 0.8)
    df.iloc[:cut].to_csv('data_train.csv', header=False, index=False)
    df.iloc[cut:].to_csv('data_valid.csv', header=False, index=False)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
