import pandas as pd
import numpy as np
import json
import os
import shutil

# 数据预处理函数部分
#任务2
def preETF1(file_path, json_file, output_folder):
    """
    Processes high-frequency trading data from a .xls or .txt file and stores it in .npy files.
    The 'thscode' from each row in .xls or each line in .txt is mapped to a new value using a JSON file,
    and data is appended to the corresponding .npy file in the 'original_dat' folder.

    :param file_path: Path to the .xls or .txt file containing the trading data.
    :param json_file: Path to the JSON file used for name extraction.
    :param output_folder: Folder where the .npy files will be saved.
    """

    # Load the JSON file for name mapping
    with open(json_file, 'r') as file:
        name_map = json.load(file)

    # Check if the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process .xls file
    if file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        a = 0
        for _, row in df.iterrows():
            a = a + 1
            if a % 1000 == 0:
                print(a)
            process_row(row, name_map, output_folder)

    # Process .txt file
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            next(file)  # Skip the header
            a = 0
            for line in file:
                # Splitting the line by comma and stripping whitespace
                data = line.strip().split(',')

                # Map the column names to the data (assuming the same order as in .xls)
                row = {
                    'thscode': data[2],
                    'latest': float(data[5]),
                    'amt': float(data[6]),
                    'vol': float(data[7])
                }
                a = a + 1
                if a % 1000 == 0:
                    print(a)
                process_row(row, name_map, output_folder)

    else:
        raise ValueError("Unsupported file format. Please provide a .xls or .txt file.")
    print("Done!")

def process_row(row, name_map, output_folder):
    """
    Process a single row of data and append it to the corresponding .npy file.

    :param row: A dictionary containing the data of a single row.
    :param name_map: A dictionary mapping thscode to a new value.
    :param output_folder: Folder where the .npy files will be saved.
    """
    thscode = row['thscode']
    mapped_name = name_map.get(thscode)

    # Continue only if a mapping exists
    if mapped_name == '503':
        file_name = f"{'503'}.npy"
        file_path = os.path.join(output_folder, file_name)
        # Extracting necessary data
        data = [row['latest']] * 4 + [row['amt'], row['vol']]
        data_array = np.array([data])

        # Load existing data if file exists, else create new array
        if os.path.isfile(file_path):
            existing_data = np.load(file_path)
            combined_data = np.vstack((existing_data, data_array))
        else:
            combined_data = data_array

        # Save the updated data
        np.save(file_path, combined_data)

def preETF2(npy_file1, npy_file2, output_file):
    """
    Processes two numpy datasets from .npy files and combines them into a new dataset,
    following specific rules for data arrangement.

    :param npy_file1: Path to the first .npy file (dataset2).
    :param npy_file2: Path to the second .npy file (original_dataset).
    :param output_file: Path for the output .npy file (dataset1).
    """

    # Load the datasets
    dataset2 = np.load(npy_file1, allow_pickle=True)
    original_dataset = np.load(npy_file2, allow_pickle=True)

    # Initialize dataset1
    dataset1 = np.zeros_like(original_dataset)

    # Copy the second column of original_dataset to the first column of dataset1
    dataset1[:, 0] = original_dataset[:, 1]

    # Constants
    rows_per_cycle = 2400
    a1 = len(dataset2)
    a2 = len(original_dataset)

    # Loop to populate dataset1
    for i in range(0, a2, rows_per_cycle):
        if i % 100 == 0:
            print(i)
        idx = i // rows_per_cycle

        if idx >= a1:
            idx = idx % a1  # Loop back to the beginning of dataset2 if idx exceeds its length

        # For the last cycle, if the number of rows is not divisible by rows_per_cycle
        end_row = min(i + rows_per_cycle, a2)

        # Update dataset1 with values from original_dataset and dataset2
        dataset1[i:end_row, 1] = original_dataset[i, 0]  # Second column of dataset1
        dataset1[i:end_row, 2:6] = dataset2[idx, 0:4]  # Third to sixth columns of dataset1

    # Save the final dataset as a .npy file
    # Adding random noise to the first column of dataset1
    random_noise = np.random.uniform(-1e-3, 1e-3, a2)  #随机噪音，避免问题
    dataset1[:, 0] += random_noise
    print(dataset1)
    np.save(output_file, dataset1)
    print("Done!")

#任务3
def preETF3(file_path, json_file, output_folder):
    """
    Processes high-frequency trading data from a .xls or .txt file and stores it in .npy files.
    The 'thscode' from each row in .xls or each line in .txt is mapped to a new value using a JSON file,
    and data is appended to the corresponding .npy file in the 'original_dat' folder.

    :param file_path: Path to the .xls or .txt file containing the trading data.
    :param json_file: Path to the JSON file used for name extraction.
    :param output_folder: Folder where the .npy files will be saved.
    """

    # Load the JSON file for name mapping
    with open(json_file, 'r') as file:
        name_map = json.load(file)

    # Check if the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process .xls file
    if file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        a = 0
        for _, row in df.iterrows():
            a = a + 1
            if a % 1000 == 0:
                print(a)
            process_row3(row, name_map, output_folder)

    # Process .txt file
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            next(file)  # Skip the header
            a = 0
            for line in file:
                # Splitting the line by comma and stripping whitespace
                data = line.strip().split(',')

                # Map the column names to the data (assuming the same order as in .xls)
                row = {
                    'thscode': data[2],
                    'latest': float(data[5]),
                    'amt': float(data[6]),
                    'vol': float(data[7])
                }
                a = a + 1
                if a % 1000 == 0:
                    print(a)
                process_row3(row, name_map, output_folder)

    else:
        raise ValueError("Unsupported file format. Please provide a .xls or .txt file.")
    print("Done!")

def process_row3(row, name_map, output_folder):
    """
    Process a single row of data and append it to the corresponding .npy file.

    :param row: A dictionary containing the data of a single row.
    :param name_map: A dictionary mapping thscode to a new value.
    :param output_folder: Folder where the .npy files will be saved.
    """
    thscode = row['thscode']
    mapped_name = name_map.get(thscode)

    # Continue only if a mapping exists
    if mapped_name:
        file_name = f"{mapped_name}.npy"
        file_path = os.path.join(output_folder, file_name)

        # Extracting necessary data
        data = [row['latest']] * 4 + [row['vol']]
        data_array = np.array([data])

        # Load existing data if file exists, else create new array
        if os.path.isfile(file_path):
            existing_data = np.load(file_path, allow_pickle=True)
            combined_data = np.vstack((existing_data, data_array))
        else:
            combined_data = data_array

        # Save the updated data
        np.save(file_path, combined_data)

def preETF4(xlsx_file, json_file, output_folder):
    """
    Process ETF trading data from an Excel file and store them in .npy files.
    Each ETF group data is mapped to a unique integer from a JSON file and saved as a numpy array.

    :param xlsx_file: Path to the Excel file containing the ETF data.
    :param json_file: Path to the JSON file with ETF name to integer mappings.
    :param output_folder: Folder where the .npy files will be saved.
    """

    # Read the Excel file
    df = pd.read_excel(xlsx_file)
    print(df.shape)

    # Load the JSON file for ETF name to integer mapping
    with open(json_file, 'r') as file:
        etf_map = json.load(file)

    # Check if the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each ETF group
    for i in range(1, df.shape[1], 5):
        print(i)
        # Extract ETF name
        etf_name = df.iloc[0, i]    #注意：这里貌似有bug！需要注意！
        print(etf_name)
        etf_int = etf_map.get(etf_name)

        # Continue only if a mapping exists
        if etf_int is not None:
            # Extract the trading data for the ETF group
            etf_data = df.iloc[3:, i:i+5].to_numpy()

            # Save the data as a .npy file
            npy_file_path = os.path.join(output_folder, f"{etf_int}.npy")
            print(npy_file_path)
            np.save(npy_file_path, etf_data)
    print("Done!")


def ETFstretch(folder_list, output_dir):
    """
    Stretches .npy files in the given folders to have the same number of rows.

    :param folder_list: List of folder paths containing the .npy files.
    :param output_dir: Directory where the stretched files will be saved.
    """

    max_rows = 0

    # Find the maximum number of rows across all .npy files
    for folder in folder_list:
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                file_path = os.path.join(folder, file)
                data = np.load(file_path, allow_pickle=True)
                max_rows = max(max_rows, len(data))
                print(f"Checked {file_path}: {len(data)} rows")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each .npy file and stretch to max_rows
    for folder in folder_list:
        output_subfolder = os.path.join(output_dir, os.path.basename(folder))
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        for file in os.listdir(folder):
            if file.endswith('.npy'):
                old_file_path = os.path.join(folder, file)
                new_file_path = os.path.join(output_subfolder, file)

                data = np.load(old_file_path, allow_pickle=True)
                a_rows = len(data)

                # Create new array with max_rows
                new_data = np.zeros((max_rows, data.shape[1]))

                # Copy first and last row
                new_data[0, :] = data[0, :]
                new_data[-1, :] = data[-1, :]

                # Distribute the remaining rows
                for i in range(1, a_rows - 1):
                    new_row_index = int(i * (max_rows - 1) / (a_rows - 1))
                    new_data[new_row_index, :] = data[i, :]

                # Interpolate missing values
                for col in range(data.shape[1]):
                    new_data[:, col] = np.interp(range(max_rows),
                                                 np.nonzero(new_data[:, col])[0],
                                                 new_data[np.nonzero(new_data[:, col])[0], col])

                # Save the new data
                np.save(new_file_path, new_data)
                print(f"Stretched and saved: {new_file_path}")

# Example usage:
#任务2-------------------------------------------------------

#从ETF的.txt/xlsx数据，生成original_dataset
#preETF1('sect_2/ETFDATA2023-11-17.txt', 'name_extracted.json', 'original_dat')
#这个得递归添加

#从ETF的original_dataset和已知的dataset2，生成dataset1
preETF2('dataset2/503.npy', 'original_dat/503.npy', 'dataset1/503.npy')
#运行一次即可

#任务3-------------------------------------------------------

#从股指期货的.txt/xlsx数据，生成origin_dom
#preETF3('sect_2/FUTUREDATA2023-11-17.txt', 'domestic_options_code_map.json', 'origin_dom')
#这个也得递归添加

#从跨境ETF的数据，生成tr_etfs_dat
preETF4('transna_data.xlsx', 'transnational_etfs_code_map.json', 'tr_etfs_dat')
#运行一次即可

#拉伸数据，求得最终值
ETFstretch(['dataset1', 'dataset2', 'origin_dom', 'original_dat', 'tr_etfs_dat'], 'endprocess')