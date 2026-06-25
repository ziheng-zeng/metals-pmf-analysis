from xact_header import headers, element_names, element_to_dl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.collections as mcoll
import pandas as pd
import pytz
import os
import glob
import re
import numpy as np

df = pd.read_csv("D:/Documents/research-2024/Xact python code/Xact_EST_May2023_July2024_old_uncert.csv")
df['TIME'] = pd.to_datetime(df['TIME'])
df.set_index('TIME', inplace=True)
df.replace(0, np.nan, inplace=True)  # Replace all zero values with NaN

# Identifying concentration and uncertainty columns
concentration_cols = [col for col in headers if " (ng/m3)" in col and "uncert" not in col.lower()]
uncertainty_cols = [col for col in headers if "uncert" in col.lower()]

# Mapping concentration columns to their corresponding uncertainty columns
metal_uncert_pairs = {}
for conc_col in concentration_cols:
    element = conc_col.split()[0]  # Assuming the first word is the element symbol
    # Find the corresponding uncertainty column, accounting for case variations
    uncert_col = next((u_col for u_col in uncertainty_cols if element in u_col), None)
    if uncert_col:
        metal_uncert_pairs[conc_col] = uncert_col

metal_uncert_pairs['S 16 (ng/m3)'] = 'S Uncert (ng/m3)' # manually change mismatch

# Apply the new uncertainty formula
for conc_col, uncert_col in metal_uncert_pairs.items():
    conc = df[conc_col].replace(0, np.nan)
    uncert = df[uncert_col].replace(0, np.nan)

    new_uncert = conc * np.sqrt((uncert / conc) ** 2 + 0.003176)

    # Overwrite the original uncertainty column
    df[uncert_col] = new_uncert.fillna(0)

    # new_col_name = conc_col.replace("(ng/m3)", "New Uncert (ng/m3)")
    #
    # df[new_col_name] = new_uncert.fillna(0)


# Print old vs. new uncertainty and percent change for selected metals
for metal in ["K", "Fe", "Al"]:
    conc_col = next((col for col in concentration_cols if col.startswith(metal + " ")), None)
    uncert_col = metal_uncert_pairs.get(conc_col)
    new_uncert_col = conc_col.replace("(ng/m3)", "New Uncert (ng/m3)")

    if conc_col and uncert_col and new_uncert_col in df.columns:
        print(f"\n--- {metal} ---")
        df_sample = df[[conc_col, uncert_col, new_uncert_col]].copy().head(5)
        df_sample.columns = ['Concentration', 'Old Uncert', 'New Uncert']
        df_sample['% Change in Uncert'] = 100 * (df_sample['New Uncert'] - df_sample['Old Uncert']) / df_sample['Old Uncert']
        print(df_sample.round(3))

df.to_csv("Xact_EST_May2023_July2024_new_uncert2.csv", index=True)
