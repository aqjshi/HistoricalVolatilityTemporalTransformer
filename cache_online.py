
import pandas as pd
import pyarrow 
import sys

parquet_file = "NEWS_20240101-142500_20251101-232422.parquet"

df = pd.read_parquet(parquet_file)


df_sorted = df.sort_values(by='time_published_ts', ascending=True)

print(df_sorted["url"].head())

