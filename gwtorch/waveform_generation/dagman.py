import os
import argparse

def generate_submit_file(memory, disk, samples, cpu, snr_dist, path="submit/short_gen.sub"):
    """Generates the HTCondor submit file for the data generation jobs."""
    content = f"""executable = /home/saqlain.afroz/miniconda3/envs/gwtorch-env/bin/generate_match_dataset_dag_gwtorch

job_name = short_gen_batch_$(ARG1)

request_memory = {memory}
request_cpus = {cpu}
request_disk = {disk}

getenv = True

log = submit/$(job_name).log
output = submit/$(job_name).output
error = submit/$(job_name).error

stream_error = True
stream_output = True

accounting_group = iucaa.gw_lens
accounting_group_user = saqlain.afroz

should_transfer_files = yes
transfer_input_files = results
when_to_transfer_output = ON_EXIT
transfer_output_files = $(ARG2)_$(ARG1), results

arguments = --num-samples {samples} --snr-prior {snr_dist} --snr-range $(ARG3) --path-name $(ARG2) --batch-size {samples / 2} --seed $(ARG1)

queue
"""
    with open(path, "w") as f:
        f.write(content)


def generate_merge_sh(path="submit/merge.sh"):
    """
    Generates the script to merge data directories and then clean them up.
    """
    script = """#!/bin/bash

# Create the central data folder and subdirectories
mkdir -p data/eccentric data/lensed data/unlensed

# Loop over all directories matching data_*
for dir in data_*; do
    # Check if it's a directory
    [ -d "$dir" ] || continue

    # Move files from subdirectories if they exist
    for sub in eccentric lensed unlensed; do
        if [ -d "$dir/$sub" ]; then
            echo "Moving files from $dir/$sub to data/$sub"
            mv "$dir/$sub"/* data/$sub/ 2>/dev/null
        fi
    done
    
    # After moving files, remove the now-empty source directory
    echo "Removing temporary directory: $dir"
    rm -r "$dir"
done

echo "All files have been moved and temporary directories are cleaned up."
"""
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)


def generate_combine_csv_sh(path="submit/combine_csv.sh"):
    """
    Generates the script to combine CSV files and then clean them up.
    """
    script = """#!/bin/bash
# Combine CSV files named data_{i}_parameters.csv into one file in order of i.

# Directory containing the CSV files
INPUT_DIR="results"
OUTPUT_FILE="./results/data_parameters.csv"

# Remove old output if exists
rm -f "$OUTPUT_FILE"

# Find files sorted by the number in their name
files=$(ls "$INPUT_DIR"/data_*_parameters.csv 2>/dev/null | sort -t_ -k2,2n)

if [ -z "$files" ]; then
    echo "No CSV files found in $INPUT_DIR"
    exit 1
fi

# Take header from the first file
head -n 1 $(echo $files | awk '{print $1}') > "$OUTPUT_FILE"

# Append data from all files (skip header in subsequent ones)
for f in $files; do
    tail -n +2 "$f" >> "$OUTPUT_FILE"
done

echo "Combined CSV written to $OUTPUT_FILE"

# Clean up the individual CSV files that have been merged
echo "Removing individual CSV files..."
rm -f $files
echo "Individual CSV files removed."
"""
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)


def generate_noop_sub(path="submit/noop.sub"):
    """Generates a dummy HTCondor job that does nothing."""
    script = """# This job does nothing and completes instantly.
# Its only purpose is to act as a synchronization point in the DAG.
universe   = vanilla
executable = /bin/true
queue
"""
    with open(path, "w") as f:
        f.write(script)


def generate_wrapper_sh(path="submit/run_local_scripts.sh"):
    """Generates a wrapper to run the two post-scripts sequentially."""
    script = """#!/bin/bash
# This script is executed on the submit node after all jobs are done.
set -e

echo "----------------------------------------------------"
echo "All data generation jobs have completed successfully."
echo "Starting post-processing on the access point..."
echo "----------------------------------------------------"

# Run the first script
echo "Executing merge.sh..."
submit/merge.sh

# Run the second script
echo "Executing combine_csv.sh..."
submit/combine_csv.sh

echo "----------------------------------------------------"
echo "Post-processing complete."
echo "----------------------------------------------------"
"""
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)


def generate_dagman(num_jobs, snr_range, path="submit/workflow.dag"):
    """Generates the corrected DAGMan file with proper dependencies."""
    lines = []
    job_names = [f"J{i}" for i in range(1, num_jobs + 1)]
    snr_list = [(snr_range[0] + i * ((snr_range[1] - snr_range[0]) / num_jobs), snr_range[0] + (i+1) * ((snr_range[1] - snr_range[0]) / num_jobs )) for i in range(num_jobs)]

    # 1. Define all data generation jobs
    for i, job_name in enumerate(job_names, 1):
        lines.append(f"JOB {job_name} submit/short_gen.sub")
        lines.append(f'VARS {job_name} ARG1 = "{i}" ARG2 = "data" ARG3 = {snr_list[i-1][0]} {snr_list[i-1][1]}')
        lines.append("")  # For readability

    # 2. Define the dummy synchronization job
    lines.append("# Dummy job to act as a synchronization point")
    lines.append("JOB ALL_DONE submit/noop.sub")
    lines.append("")

    # 3. Define the dependencies
    parent_line = " ".join(job_names)
    lines.append("# The ALL_DONE job runs only after all data jobs are complete")
    lines.append(f"PARENT {parent_line} CHILD ALL_DONE")
    lines.append("")

    # 4. Define the post-script to run on the access point
    lines.append("# This script runs on the access point after ALL_DONE finishes")
    lines.append("SCRIPT POST ALL_DONE submit/run_local_scripts.sh")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a complete DAGMan workflow with helper scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required argument
    parser.add_argument("--jobs", type=int, required=True,
                        help="Number of jobs to include in the DAGMan file")
    
    # Optional arguments for job resources
    parser.add_argument("--memory", type=str, default="20G",
                        help="Memory request for each job (e.g., '8G', '1024M')")
    parser.add_argument("--disk", type=str, default="20G",
                        help="Disk request for each job (e.g., '40G', '2048M')")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to generate per job")
    parser.add_argument("--cpu", type=int, default=5,
                        help="Number of CPUs to request for each job")
    parser.add_argument("--snr-dist", type=str, choices=['uniform', 'powerlaw'], default='powerlaw',
                        help='Defines the prior distribution for sampling in SNR. Default is PowerLaw.')
    parser.add_argument("--snr-min", type=int, default=8,
                        help='Minimum of SNR values to sample from. Default is 8.')
    parser.add_argument("--snr-max", type=int, default=100,
                        help='Maximum of SNR values to sample from. Default is 100.')
    
    args = parser.parse_args()
    
    # Create directories needed for the workflow
    os.makedirs("submit", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Generate all necessary files
    generate_submit_file(
        memory=args.memory,
        disk=args.disk,
        samples=args.samples,
        cpu=args.cpu,
        snr_dist=args.snr_dist
    )
    generate_merge_sh()
    generate_combine_csv_sh()
    generate_noop_sub()
    generate_wrapper_sh()
    generate_dagman(num_jobs=args.jobs, snr_range=(args.snr_min, args.snr_max))

    print(f"✅ Successfully generated workflow for {args.jobs} jobs in the 'submit' directory.")
    print("To run, execute: condor_submit_dag submit/workflow.dag")


if __name__ == "__main__":
    main()