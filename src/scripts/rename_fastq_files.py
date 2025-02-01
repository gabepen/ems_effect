def rename_files(directory):
    # Find all R1 and R2 FASTQ files in the directory
    fastq_files = glob.glob(os.path.join(directory, "*.fastq.gz"))

    for file_path in fastq_files:
        file_name = os.path.basename(file_path)
        
        # Find the character after the '-'
        dash_index = file_name.find('-')
        if dash_index == -1 or dash_index + 1 >= len(file_name):
            print(f"Warning: Unexpected file name format {file_name}. Skipping.")
            continue
        
        char_after_dash = file_name[dash_index + 1]
        
        # Remove the character after the '-' and the '-'
        new_file_name = file_name[:dash_index] + file_name[dash_index + 2:]
        
        # Insert the character with a '_' before the first '.'
        dot_index = new_file_name.find('.')
        new_file_name = new_file_name[:dot_index - 7] + f"_{char_after_dash}" + new_file_name[dot_index - 7:]
        
        # Construct the new file path
        new_file_path = os.path.join(directory, new_file_name)
        
        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"Renamed {file_name} to {new_file_name}")
        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_fastq_files.py <input_dir> ")
        sys.exit(1)

    input_dir = sys.argv[1]

    rename_files(input_dir)