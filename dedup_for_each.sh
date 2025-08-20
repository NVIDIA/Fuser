#!/bin/bash

# Check for correct number of arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <input_folder> <output_folder>"
  exit 1
fi

input_folder="$1"
output_folder="$2"

# Create output folder if it doesn't exist
mkdir -p "$output_folder"

# Process each file in the input folder
for input_file in "$input_folder"/*; do
  # Ensure it's a file
  [ -f "$input_file" ] || continue

  # Extract just the filename
  filename=$(basename "$input_file")
  output_file="$output_folder/$filename"

  # Process the file
  sort "$input_file" | uniq -c | awk '{count=$1; $1=""; sub(/^ /, ""); print $0 " " count}' > "$output_file"
done

echo "Processing complete. Output saved in '$output_folder'."
