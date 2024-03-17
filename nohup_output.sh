#!/usr/bin/env bash


# Check if file path is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

file="$1"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File '$file' does not exist."
    exit 1
fi

# Infinite loop to continuously print the last line of the file
while true; do
    # Print the last line of the file
    tail -n 1 "$file"
    # Wait for 1 second
    sleep 1
done
