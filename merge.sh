#!/bin/bash

# Create the central data folder and subdirectories
mkdir -p data/eccentric data/lensed data/unlensed

# Loop over all directories matching data_*
for dir in data_*; do
    # Check if it's a directory
    [ -d "$dir" ] || continue

    # Move files from eccnetric, lensed, unlensed if they exist
    for sub in eccentric lensed unlensed; do
        if [ -d "$dir/$sub" ]; then
            echo "Moving files from $dir/$sub to data/$sub"
            mv "$dir/$sub"/* data/$sub/ 2>/dev/null
        fi
    done
done

echo "All files have been moved."