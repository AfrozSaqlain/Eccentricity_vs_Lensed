#!/bin/bash
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