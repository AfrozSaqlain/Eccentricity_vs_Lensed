#!/bin/bash

# Source and target directories
SRC_DIR="data_2"
DEST_DIR="test_folder"

# Create destination subdirectories
mkdir -p "$DEST_DIR/eccentric" "$DEST_DIR/lensed" "$DEST_DIR/unlensed"

# Loop through each class folder
for class in eccentric lensed unlensed; do
    echo "Processing class: $class"

    # List all files in class folder
    files=("$SRC_DIR/$class"/*)

    # Total number of files
    total=${#files[@]}

    if [ "$total" -eq 0 ]; then
        echo "No files in $class, skipping..."
        continue
    fi

    # Calculate 30% of files (rounded down)
    num_to_move=$(( total * 30 / 100 ))

    echo "Moving $num_to_move out of $total files from $class..."

    # Shuffle and select 30%
    shuf -e "${files[@]}" | head -n "$num_to_move" | while read -r file; do
        mv "$file" "$DEST_DIR/$class/"
    done
done

echo "Done! Files moved to $DEST_DIR"

