#!/bin/bash

# Define the Python script to run
python_script="build_subdatasets.py"

# Function to run the Python script and retry if it fails
until nohup python "$python_script"; do
    echo "Script failed. Retrying in 3 seconds..."
    sleep 3
done

