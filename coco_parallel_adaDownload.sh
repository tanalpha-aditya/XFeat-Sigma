#!/bin/bash
#SBATCH --account=neuro
#SBATCH --partition=ihub
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:0
#SBATCH --mem=25000
#SBATCH --time=4-00:00:00
#SBATCH --output=download_coco_93.txt
#SBATCH --nodelist=gnode093

# Maximum number of retries
MAX_RETRIES=300  
RETRY_COUNT=0  

# COCO dataset URLs
URLS=(
    "http://images.cocodataset.org/zips/train2017.zip"
    "http://images.cocodataset.org/zips/val2017.zip"
    "http://images.cocodataset.org/zips/test2017.zip"
)

# Directory to save the dataset
SCRATCH_DIR="/scratch/narasimha.pai/coco"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

for URL in "${URLS[@]}"; do
    FILE_NAME=$(basename "$URL")
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "Attempt #$((RETRY_COUNT + 1)) to download $FILE_NAME"

        # Run aria2c with resume support
        aria2c -c -x 16 -s 16 -k 256M "$URL" -d "$SCRATCH_DIR" -o "$FILE_NAME"

        # Check exit status of aria2c
        if [ $? -eq 0 ]; then
            echo "$FILE_NAME downloaded successfully!"
            break  # Exit loop if download is successful
        else
            echo "Download failed for $FILE_NAME! Retrying in 30 seconds..."
            ((RETRY_COUNT++))
            sleep 30  # Wait before retrying
        fi
    done

    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Max retries reached for $FILE_NAME. Skipping..."
    fi
    RETRY_COUNT=0  # Reset retry count for next file

done