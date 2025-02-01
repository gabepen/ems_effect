import os
import sys
import csv
import shutil

'''This script can be used to collect raw reads from an EMS data folder and copy them to a new folder with new names
based on the sample ID instead of the sequencing acession.
'''

def copy_fastq_files(data_folder: str, output_folder: str):
    
    # Path to the association table CSV
    association_table_path = os.path.join(data_folder, 'barcodeAssociationTable.txt')

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the association table
    with open(association_table_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            accession_id = row['AccessionID']
            sample_id = row['ClientAccessionID']

            # Path to the subfolder
            subfolder_path = None
            for folder_name in os.listdir(data_folder):
                if folder_name.startswith(accession_id):
                    subfolder_path = os.path.join(data_folder, folder_name)
                    break

            if subfolder_path is None:
                print(f"Warning: No subfolder starting with {accession_id} found in {data_folder}.")
                continue

            # Check if the subfolder exists
            if not os.path.exists(subfolder_path):
                print(f"Warning: Subfolder {subfolder_path} does not exist.")
                continue

            # Find the FASTQ files in the subfolder
            fastq_files = [f for f in os.listdir(subfolder_path) if f.endswith('.fastq.gz')]

            # Check if there are exactly two FASTQ files
            if len(fastq_files) != 2:
                print(f"Warning: Expected 2 FASTQ files in {subfolder_path}, found {len(fastq_files)}.")
                continue

            # Copy the FASTQ files to the output folder with new names
            for fastq_file in fastq_files:
                if '_R1_' in fastq_file:
                    new_file_name = f"{sample_id}_R1.fastq.gz"
                elif '_R2_' in fastq_file:
                    new_file_name = f"{sample_id}_R2.fastq.gz"
                else:
                    print(f"Warning: Unexpected FASTQ file name {fastq_file} in {subfolder_path}.")
                    continue

                src_path = os.path.join(subfolder_path, fastq_file)
                dest_path = os.path.join(output_folder, new_file_name)
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <data_folder> <output_folder>")
        sys.exit(1)

    data_folder = sys.argv[1]
    output_folder = sys.argv[2]

    copy_fastq_files(data_folder, output_folder)