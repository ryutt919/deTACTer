import pandas as pd

encodings = ['utf-8', 'cp949', 'utf-8-sig', 'euc-kr']
path = 'c:/Users/Public/Documents/DIK/deTACTer/data/raw_data.csv'

for enc in encodings:
    try:
        df = pd.read_csv(path, encoding=enc, nrows=5)
        print(f"\n--- Result for {enc} ---")
        print(df[['player_name_ko', 'team_name_ko']].head())
    except Exception as e:
        print(f"\n--- Failed for {enc} ---")
        print(e)
