import pandas as pd

# === 1. Load your file ===
file_path = "Xact_EST_May2023_July2025_combined.csv"   # change to your actual file path
df = pd.read_csv(file_path)

# === 2. Convert TIME column to UTC ===
# First parse the TIME column (it looks like it already has offsets, e.g. -04:00)
df['TIME'] = pd.to_datetime(df['TIME'], utc=True)

# Convert everything explicitly to UTC
df['TIME'] = df['TIME'].dt.tz_convert('UTC')

# (Optional) If you want to drop the "+00:00" timezone info and keep plain strings
df['TIME'] = df['TIME'].dt.strftime("%Y-%m-%d %H:%M:%S")

# === 3. Save the updated file ===
output_path = "Xact_UTC_May2023_July2025.csv"
df.to_csv(output_path, index=False)

print(f"Saved UTC-converted file to {output_path}")
