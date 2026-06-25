import glob
import pandas as pd
import numpy as np
import os
from datetime import datetime


def process_file_chunk(file_path, mid_D, dlogDp, density=1.4, apply_stp=True):
    """Process a single file and return concentrations"""
    try:
        # Load single file
        df = pd.read_csv(file_path, skiprows=52)

        # Set datetime index
        tsdf = df.set_index('DateTime Sample Start')

        # Extract size distribution data (columns 41:425)
        artsdf = np.array(tsdf)
        dNdlogDp = artsdf[:, 41:425]

        # STP conversion factor if requested
        stp_factor = 1.0
        if apply_stp:
            try:
                # Extract sheath pressure (kPa) and sheath temperature (C) if available
                sheath_pressure = artsdf[:, tsdf.columns.get_loc('Sheath Pressure (kPa)')]
                sheath_temp = artsdf[:, tsdf.columns.get_loc('Sheath Temp (C)')]

                # STP conversion: (101.35/P) * ((273.15+T)/273.15)
                stp_factor = (101.35 / sheath_pressure) * ((273.15 + sheath_temp) / 273.15)
                print(f"  STP conversion applied for {os.path.basename(file_path)}")
            except (KeyError, ValueError):
                print(f"  Warning: STP data not found in {os.path.basename(file_path)}, using ambient conditions")
                stp_factor = 1.0

        # Apply STP correction to size distribution
        if isinstance(stp_factor, np.ndarray):
            dNdlogDp_stp = dNdlogDp * stp_factor[:, np.newaxis]
        else:
            dNdlogDp_stp = dNdlogDp * stp_factor

        # Calculate number concentration
        dN = dNdlogDp_stp * dlogDp
        N = np.nansum(dN, axis=1)

        # Calculate mass concentration
        dMdlogDp = (density / 1e9) * (np.pi / 6.) * mid_D ** 3 * dNdlogDp_stp
        dM = dMdlogDp * dlogDp
        M = np.nansum(dM, axis=1)

        # Create results DataFrame for this file
        results = pd.DataFrame({
            'DateTime': tsdf.index,
            'Number_Concentration_per_cm3': N,
            'Mass_Concentration_ug_per_m3': M
        })

        return results

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def main():
    print("SMPS Data Processing - Memory-Efficient Version with STP Conversion")
    print("=" * 70)

    ### 1. Setup ###
    path = r"D:\Documents\research-2024\SMPS data\data-all-time"
    csv_files = glob.glob(os.path.join(path, 'SMPS*.csv'))

    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        print("No SMPS*.csv files found.")
        return

    # Ask user about STP conversion
    apply_stp = True  # Set to False if you want ambient conditions
    print(f"STP conversion: {'ENABLED' if apply_stp else 'DISABLED'}")
    print("(Concentrations will be corrected to Standard Temperature and Pressure)")
    print()

    ### 2. Get size bin information from first file ###
    print("Reading size bin information from first file...")

    try:
        first_df = pd.read_csv(csv_files[0], skiprows=52)
        first_tsdf = first_df.set_index('DateTime Sample Start')

        # Extract size information
        mid_D = np.array([float(x) for x in first_tsdf.columns[41:425]])

        # Calculate bin boundaries
        avg_diff = np.mean(np.diff(np.log10(mid_D)))
        D_bound = np.full(mid_D.shape[0] + 1, np.nan)

        for i in range(1, len(D_bound) - 1):
            D_bound[i] = 10 ** (0.5 * (np.log10(mid_D[i]) + np.log10(mid_D[i - 1])))

        D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5 * avg_diff)
        D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5 * avg_diff)

        D_low = D_bound[0:-1]
        D_high = D_bound[1:]
        dlogDp = np.log10(D_high) - np.log10(D_low)

        print(f"Size range: {mid_D[0]:.1f} - {mid_D[-1]:.1f} nm")
        print(f"Number of size bins: {len(mid_D)}")

        # Clean up
        del first_df, first_tsdf

    except Exception as e:
        print(f"Error reading size information: {e}")
        return

    ### 3. Process files in chunks ###
    print(f"\nProcessing {len(csv_files)} files...")

    chunk_size = 20  # Process 20 files at a time
    all_results = []

    for i in range(0, len(csv_files), chunk_size):
        chunk_files = csv_files[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(csv_files) + chunk_size - 1) // chunk_size

        print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_files)} files)...")

        chunk_results = []
        for j, file_path in enumerate(chunk_files):
            print(f"  File {i + j + 1}/{len(csv_files)}: {os.path.basename(file_path)}")

            result = process_file_chunk(file_path, mid_D, dlogDp, apply_stp=apply_stp)
            if result is not None:
                chunk_results.append(result)

        # Combine chunk results
        if chunk_results:
            chunk_combined = pd.concat(chunk_results, ignore_index=True)
            all_results.append(chunk_combined)
            print(f"  Chunk {chunk_num} completed: {len(chunk_combined)} records")

        # Clean up chunk data to free memory
        del chunk_results
        if 'chunk_combined' in locals():
            del chunk_combined

    ### 4. Combine all results ###
    print("\nCombining all results...")

    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        # Sort by datetime - handle European date format (DD/MM/YYYY)
        final_results['DateTime'] = pd.to_datetime(final_results['DateTime'], dayfirst=True, format='mixed')
        final_results = final_results.sort_values('DateTime').reset_index(drop=True)

        print(f"Total records processed: {len(final_results)}")

        ### 5. Save results ###
        output_path = r'D:\Documents'
        stp_suffix = "_STP" if apply_stp else "_ambient"
        output_filename = f'SMPS_UTC_mass_concentrations_alltime{stp_suffix}.csv'

        os.makedirs(output_path, exist_ok=True)
        full_output_path = os.path.join(output_path, output_filename)

        # Save with datetime formatting
        final_results_save = final_results.copy()
        final_results_save['DateTime'] = final_results_save['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        final_results_save.to_csv(full_output_path, index=False)

        print(f"\nData processing complete!")
        print(f"Results saved to: {full_output_path}")

        # Show summary statistics
        print(f"\nDate range: {final_results['DateTime'].min()} to {final_results['DateTime'].max()}")
        print(f"\nSummary statistics:")
        print(final_results[['Number_Concentration_per_cm3', 'Mass_Concentration_ug_per_m3']].describe())

        # Show first and last few rows
        print(f"\nFirst 5 rows:")
        print(final_results.head())
        print(f"\nLast 5 rows:")
        print(final_results.tail())

    else:
        print("No data could be processed successfully.")


if __name__ == "__main__":
    main()
#
# import glob
# import pandas as pd
# import os
# from datetime import datetime
#
#
# def main():
#     print("SMPS Raw Data Concatenation - Memory-Efficient Version")
#     print("=" * 60)
#
#     ### 1. Setup ###
#     path = r"D:\Documents\research-2024\SMPS data\data-all-time"
#     csv_files = glob.glob(os.path.join(path, 'SMPS*.csv'))
#
#     print(f"Found {len(csv_files)} CSV files")
#
#     if len(csv_files) == 0:
#         print("No SMPS*.csv files found.")
#         return
#
#     ### 2. Process files in chunks to manage memory ###
#     print(f"\nConcatenating {len(csv_files)} files...")
#
#     chunk_size = 15  # Process fewer files at once for raw data (larger files)
#     all_data_chunks = []
#
#     # Read first file to get column structure
#     print("Reading first file to establish column structure...")
#     try:
#         first_df = pd.read_csv(csv_files[0], skiprows=52)
#         expected_columns = first_df.columns.tolist()
#         print(f"Expected columns: {len(expected_columns)}")
#         print(f"First few columns: {expected_columns[:5]}")
#         print(f"Last few columns: {expected_columns[-5:]}")
#         all_data_chunks.append(first_df)
#         print(f"First file loaded: {len(first_df)} rows")
#         del first_df
#     except Exception as e:
#         print(f"Error reading first file: {e}")
#         return
#
#     # Process remaining files in chunks
#     remaining_files = csv_files[1:]  # Skip first file since we already processed it
#
#     for i in range(0, len(remaining_files), chunk_size):
#         chunk_files = remaining_files[i:i + chunk_size]
#         chunk_num = i // chunk_size + 1
#         total_chunks = (len(remaining_files) + chunk_size - 1) // chunk_size
#
#         print(f"\nProcessing chunk {chunk_num}/{total_chunks} ({len(chunk_files)} files)...")
#
#         chunk_data = []
#         for j, file_path in enumerate(chunk_files):
#             file_num = i + j + 2  # +2 because we start from file 2 (first was processed separately)
#             print(f"  File {file_num}/{len(csv_files)}: {os.path.basename(file_path)}")
#
#             try:
#                 df = pd.read_csv(file_path, skiprows=52)
#
#                 # Check if columns match expected structure
#                 if len(df.columns) != len(expected_columns):
#                     print(f"    Warning: Column count mismatch in {os.path.basename(file_path)} "
#                           f"(expected {len(expected_columns)}, got {len(df.columns)})")
#
#                 # Ensure column consistency
#                 df = df.reindex(columns=expected_columns)
#                 chunk_data.append(df)
#                 print(f"    Loaded: {len(df)} rows")
#
#             except Exception as e:
#                 print(f"    Error loading {os.path.basename(file_path)}: {e}")
#                 continue
#
#         # Combine chunk data
#         if chunk_data:
#             chunk_combined = pd.concat(chunk_data, ignore_index=True)
#             all_data_chunks.append(chunk_combined)
#             print(f"  Chunk {chunk_num} completed: {len(chunk_combined)} total rows")
#
#             # Clean up memory
#             del chunk_data, chunk_combined
#
#     ### 3. Combine all chunks ###
#     print(f"\nCombining all {len(all_data_chunks)} chunks...")
#
#     if all_data_chunks:
#         # Combine all chunks into final dataset
#         final_data = pd.concat(all_data_chunks, ignore_index=True)
#         print(f"Total records in combined dataset: {len(final_data)}")
#         print(f"Total columns: {len(final_data.columns)}")
#
#         # Clean up chunk data to free memory
#         del all_data_chunks
#
#         ### 4. Save raw concatenated data ###
#         output_path = r'D:\Documents'
#         output_filename = 'SMPS_raw_data_concatenated_alltime.csv'
#
#         os.makedirs(output_path, exist_ok=True)
#         full_output_path = os.path.join(output_path, output_filename)
#
#         print(f"\nSaving concatenated raw data...")
#         print(f"Output file: {full_output_path}")
#
#         # Save the raw concatenated data
#         final_data.to_csv(full_output_path, index=False)
#
#         print(f"\nRaw data concatenation complete!")
#         print(f"File saved to: {full_output_path}")
#
#         # Show dataset info
#         if 'DateTime Sample Start' in final_data.columns:
#             # Try to parse dates to show date range
#             try:
#                 dates = pd.to_datetime(final_data['DateTime Sample Start'], dayfirst=True, format='mixed')
#                 print(f"\nDate range: {dates.min()} to {dates.max()}")
#             except Exception as e:
#                 print(f"Could not parse dates for range display: {e}")
#
#         print(f"\nDataset summary:")
#         print(f"- Total rows: {len(final_data):,}")
#         print(f"- Total columns: {len(final_data.columns)}")
#         print(f"- File size: ~{os.path.getsize(full_output_path) / (1024 ** 2):.1f} MB")
#
#         # Show first few rows
#         print(f"\nFirst 3 rows (first 5 columns):")
#         print(final_data.iloc[:3, :5])
#
#         # Show column names
#         print(f"\nColumn names (first 10):")
#         for i, col in enumerate(final_data.columns[:10]):
#             print(f"  {i + 1}: {col}")
#         if len(final_data.columns) > 10:
#             print(f"  ... and {len(final_data.columns) - 10} more columns")
#
#     else:
#         print("No data could be concatenated successfully.")
#
#
# if __name__ == "__main__":
#     main()