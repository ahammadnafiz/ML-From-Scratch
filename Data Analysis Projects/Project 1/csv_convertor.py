import json
import csv as csv_module
import os
from glob import glob

def json_to_csv(input_pattern, output_file):
    # List to store all data
    all_data = []

    # Read all JSON files
    for json_file in glob(input_pattern):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except json.JSONDecodeError:
                print(f"Error reading {json_file}. Skipping this file.")

    if not all_data:
        print("No valid data found in JSON files.")
        return

    # Specify the columns we want in our CSV
    columns = ['job_title', 'city', 'country', 'company_name', 'is_remote', 'date']

    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv_module.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for item in all_data:
                writer.writerow({col: item.get(col, '') for col in columns})
        print(f"CSV file '{output_file}' has been created successfully.")
    except PermissionError:
        print(f"Error: Unable to write to '{output_file}'. Make sure you have write permissions and the file is not open in another program.")

# Usage
input_pattern = '*.json'  # This will match all JSON files in the current directory
output_file = 'combined_jobs.csv'

json_to_csv(input_pattern, output_file)