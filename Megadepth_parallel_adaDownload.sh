#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:0
#SBATCH --mem=75000
#SBATCH --time=4-00:00:00
#SBATCH --output=download2.txt
#SBATCH --nodelist=gnode076

# Maximum number of retries (set to a large number or remove the check for infinite retries)
MAX_RETRIES=300  
RETRY_COUNT=0  

# URL to download
URL="https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz"

# Path to save the file
OUTPUT_FILE="MegaDepth_v1.tar.gz"

cd /scratch/yash9439

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt #$((RETRY_COUNT + 1)) to download $URL"

    # Run aria2c with resume support
    aria2c -c -x 16 -s 16 -k 256M "$URL" -d . -o "$OUTPUT_FILE"

    # Check exit status of aria2c
    if [ $? -eq 0 ]; then
        echo "Download completed successfully!"
        break  # Exit loop if download is successful
    else
        echo "Download failed! Retrying in 30 seconds..."
        ((RETRY_COUNT++))
        sleep 30  # Wait before retrying
    fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "Max retries reached. Exiting..."
fi

