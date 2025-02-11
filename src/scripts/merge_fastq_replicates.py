import os
import sys
import glob
import gzip
import shutil

def merge_fastq_files_timepoints(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all unique sample prefixes (assuming format: sample_X_R1.fastq.gz)
    sample_prefixes = set()
    for file in glob.glob(os.path.join(input_dir, "*_R1.fastq.gz")):
        sample_prefix = os.path.basename(file).split('_R1.fastq.gz')[0]
        
        # hard code for specific sample names from 110420 sequencing
        #sample_prefix = sample_prefix.replace('_Rd2', '')
        #sample_prefix = sample_prefix.replace('_ReSeq', '')
        
        # hard code for sample names from 280120 and 180220 sequencing
        sample_prefix = sample_prefix.split('-')[0]
        
        #sample_prefix = sample_prefix[:-2]
        sample_prefixes.add(sample_prefix)
    
    # sanity check for string parsing 
    print(sample_prefixes)
    input()
    
    # Merge the FASTQ files for each sample
    for sample_prefix in sample_prefixes:
        #r1_files = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*_R1.fastq.gz")))
        #r2_files = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*_R2.fastq.gz")))

        r1_files_3d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*3d*_R1.fastq.gz")))
        r2_files_3d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*3d*_R2.fastq.gz")))
        
        r1_files_7d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*7d*_R1.fastq.gz")))
        r2_files_7d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*7d*_R2.fastq.gz")))
        
        
        if len(r1_files_3d) != len(r2_files_3d):
            print(f"Warning: Mismatched number of R1 and R2 files for sample {sample_prefix}. Skipping.")
            continue

        merged_r1_path_3d = os.path.join(output_dir, f"{sample_prefix}_3d_R1.fastq.gz")
        merged_r2_path_3d = os.path.join(output_dir, f"{sample_prefix}_3d_R2.fastq.gz")

        merged_r1_path_7d = os.path.join(output_dir, f"{sample_prefix}_7d_R1.fastq.gz")
        merged_r2_path_7d = os.path.join(output_dir, f"{sample_prefix}_7d_R2.fastq.gz")
        
        # Merge R1 files
        if len(r1_files_3d) != 0:
            with open(merged_r1_path_3d, 'wb') as merged_r1:
                for r1_file in r1_files_3d:
                    with open(r1_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r1)
        if len(r1_files_7d) != 0:         
            with open(merged_r1_path_7d, 'wb') as merged_r1:
                for r1_file in r1_files_7d:
                    with open(r1_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r1)

        # Merge R2 files
        if len(r2_files_3d) != 0:
            with open(merged_r2_path_3d, 'wb') as merged_r2:
                for r2_file in r2_files_3d:
                    with open(r2_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r2)
        if len(r2_files_7d) != 0:    
            with open(merged_r2_path_7d, 'wb') as merged_r2:
                for r2_file in r2_files_7d:
                    with open(r2_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r2)

        #print(f"Merged {len(r1_files)} R1 files and {len(r2_files)} R2 files for sample {sample_prefix}.")

        #print(f"Merged {len(r1_files)} R1 files and {len(r2_files)} R2 files for sample {sample_prefix}.")

def merge_fastq_files(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all unique sample prefixes (assuming format: sample_X_R1.fastq.gz)
    sample_prefixes = set()
    for file in glob.glob(os.path.join(input_dir, "*_R1.fastq.gz")):
        sample_prefix = os.path.basename(file).split('_R1.fastq.gz')[0]
        
        # hard code for specific sample names from 110420 sequencing
        #sample_prefix = sample_prefix.replace('_Rd2', '')
        #sample_prefix = sample_prefix.replace('_ReSeq', '')
        
        # hard code for sample names from 280120 and 180220 sequencing
        sample_prefix = sample_prefix.split('-')[0]
        
        #sample_prefix = sample_prefix[:-2]
        sample_prefixes.add(sample_prefix)
    
    # sanity check for string parsing 
    print(sample_prefixes)
    input()
    
    # Merge the FASTQ files for each sample
    for sample_prefix in sample_prefixes:
        #r1_files = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*_R1.fastq.gz")))
        #r2_files = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*_R2.fastq.gz")))

        r1_files_3d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*3d*_R1.fastq.gz")))
        r2_files_3d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*3d*_R2.fastq.gz")))
        
        r1_files_7d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*7d*_R1.fastq.gz")))
        r2_files_7d = sorted(glob.glob(os.path.join(input_dir, f"{sample_prefix}*7d*_R2.fastq.gz")))
        
        
        if len(r1_files_3d) != len(r2_files_3d):
            print(f"Warning: Mismatched number of R1 and R2 files for sample {sample_prefix}. Skipping.")
            continue

        merged_r1_path_3d = os.path.join(output_dir, f"{sample_prefix}_3d_R1.fastq.gz")
        merged_r2_path_3d = os.path.join(output_dir, f"{sample_prefix}_3d_R2.fastq.gz")

        merged_r1_path_7d = os.path.join(output_dir, f"{sample_prefix}_7d_R1.fastq.gz")
        merged_r2_path_7d = os.path.join(output_dir, f"{sample_prefix}_7d_R2.fastq.gz")
        
        # Merge R1 files
        if len(r1_files_3d) != 0:
            with open(merged_r1_path_3d, 'wb') as merged_r1:
                for r1_file in r1_files_3d:
                    with open(r1_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r1)
        if len(r1_files_7d) != 0:         
            with open(merged_r1_path_7d, 'wb') as merged_r1:
                for r1_file in r1_files_7d:
                    with open(r1_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r1)

        # Merge R2 files
        if len(r2_files_3d) != 0:
            with open(merged_r2_path_3d, 'wb') as merged_r2:
                for r2_file in r2_files_3d:
                    with open(r2_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r2)
        if len(r2_files_7d) != 0:    
            with open(merged_r2_path_7d, 'wb') as merged_r2:
                for r2_file in r2_files_7d:
                    with open(r2_file, 'rb') as f:
                        shutil.copyfileobj(f, merged_r2)

        #print(f"Merged {len(r1_files)} R1 files and {len(r2_files)} R2 files for sample {sample_prefix}.")

        #print(f"Merged {len(r1_files)} R1 files and {len(r2_files)} R2 files for sample {sample_prefix}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_fastq.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    merge_fastq_files(input_dir, output_dir)