import pandas as pd

# Load both files
file1_path = "Xact_EST_May2023_July2025_combined.csv"
file2_path = "Xact_July_to_Oct2025.csv"

# Load May 2023 – July 2024 data
df1 = pd.read_csv(file1_path, parse_dates=["TIME"])
df1.set_index("TIME", inplace=True)

# Load Aug 2024 – May 2025 data
df2 = pd.read_csv(file2_path, parse_dates=["TIME"])
df2.set_index("TIME", inplace=True)

# Merge and sort two datafiles
df_combined = pd.concat([df1, df2])
df_combined.sort_index(inplace=True)

# print(df_combined.head())
# print(df_combined.columns.tolist())

output_path = "D:/Documents/research-2024/Xact python code/Xact_EST_May2023_Oct2025_combined.csv"
df_combined.to_csv(output_path)
