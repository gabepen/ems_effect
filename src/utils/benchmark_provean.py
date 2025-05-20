import subprocess
import psutil
import time
import threading
from statistics import mean
import os

def run_provean(provean_cmd):
    """Run the PROVEAN command."""
    process = subprocess.Popen(provean_cmd, shell=True)
    return process

def get_all_children(pid):
    """Get all child processes of a given PID."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        return [parent] + children
    except psutil.NoSuchProcess:
        return []

def monitor_cpu_usage(pid, cpu_usages, interval=1):
    """Monitor CPU usage of a process and all its children."""
    try:
        while True:
            # Get all child processes
            processes = get_all_children(pid)
            if not processes:
                break  # No processes to monitor
                
            # Calculate total CPU usage across all processes
            total_cpu_time = 0
            for proc in processes:
                try:
                    # Get CPU times for this process
                    cpu_times = proc.cpu_times()
                    total_cpu_time += (cpu_times.user + cpu_times.system)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            # Append the total CPU time
            cpu_usages.append(total_cpu_time)
            
            # Sleep for the interval
            time.sleep(interval)
    except Exception as e:
        print(f"Error in monitoring: {e}")

def main():
    # Example PROVEAN command
    provean_cmd = (
        "/bin/bash /storage1/gabe/ems_effect_code/provean/provean.sh -q /storage1/gabe/ems_effect_code/data/provean_sets/29554713.fa "
        "-v /storage1/gabe/ems_effect_code/effect_outputs/full_run_1_Ncomp_pergene/provean_files/29554713_vars/29554713.var "
        "--supporting_set /storage1/gabe/ems_effect_code/data/provean_sets/29554713.sss  --num_threads 1"
    )

    # Start timing
    start_time = time.time()

    # Start PROVEAN
    process = run_provean(provean_cmd)

    # Shared list to store CPU usage data
    cpu_usages = []

    # Start monitoring CPU usage in a separate thread
    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(process.pid, cpu_usages))
    monitor_thread.daemon = True  # Make thread daemon so it exits when main thread exits
    monitor_thread.start()

    # Wait for PROVEAN to finish
    process.wait()

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Calculate CPU usage metrics
    if cpu_usages:
        # Calculate the difference between consecutive measurements
        cpu_usage_diffs = [cpu_usages[i] - cpu_usages[i-1] for i in range(1, len(cpu_usages))]
        # Average CPU usage per second
        avg_cpu_per_second = mean(cpu_usage_diffs) if cpu_usage_diffs else 0
        # Estimate threads used (CPU seconds per real second)
        estimated_threads = avg_cpu_per_second
    else:
        avg_cpu_per_second = 0
        estimated_threads = 0

    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total CPU time: {cpu_usages[-1] if cpu_usages else 0:.2f} seconds")
    print(f"Average CPU usage per second: {avg_cpu_per_second:.2f}")
    print(f"Estimated threads used: {estimated_threads:.2f}")

if __name__ == "__main__":
    main()