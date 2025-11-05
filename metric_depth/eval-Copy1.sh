# Define encoder configurations with their batch sizes
declare -A BATCH_SIZES
BATCH_SIZES["vits"]=1
BATCH_SIZES["vitb"]=1
BATCH_SIZES["vitl"]=1

# --- Define associative array to store LOAD_FROM paths as space-separated strings ---
# Bash associative array values must be strings. We'll store lists as strings,
# then convert back to arrays.
declare -A LOAD_FROM_PATH_STRINGS

# Paths for 'vits' (joined by spaces into a single string)
LOAD_FROM_PATH_STRINGS["vits"]="/data/depthanything/depth_anything_v2_metric_vkitti_vits.pth \
/data/depthanything/depth_anything_v2_metric_hypersim_vits.pth \
/data/train_combined_with_dino_20250506_235908/best_abs_rel.pth \
/data/train_combined/best_abs_rel.pth \
/data/train_combined_20250506_062322_2to1_full/best_abs_rel.pth \
/data/train_combined_20250506_000932_2to1_headonly/best_abs_rel.pth"

# Paths for 'vitb' (joined by spaces into a single string)
LOAD_FROM_PATH_STRINGS["vitb"]="/data/depthanything/depth_anything_v2_metric_vkitti_vitb.pth \
/data/depthanything/depth_anything_v2_metric_hypersim_vitb.pth"

# Paths for 'vitl' (joined by spaces into a single string)
LOAD_FROM_PATH_STRINGS["vitl"]="/data/depthanything/depth_anything_v2_metric_vkitti_vitl.pth \
/data/depthanything/depth_anything_v2_metric_hypersim_vitl.pth"

# Loop through each defined encoder
for ENCODER in "${!BATCH_SIZES[@]}"; do
    BATCH_SIZE="${BATCH_SIZES[$ENCODER]}"
    BASE_SAVE_PATH="/data/eval/dam/inhouse/$ENCODER"

    echo "=== Starting evaluations for ENCODER: $ENCODER with BATCH_SIZE: $BATCH_SIZE ==="

    # Check if path strings are defined for this encoder
    if [[ -v LOAD_FROM_PATH_STRINGS[$ENCODER] ]]; then
        # Retrieve the single string containing all paths for the current encoder
        CURRENT_PATHS_STRING="${LOAD_FROM_PATH_STRINGS[$ENCODER]}"

        # Convert the space-separated string back into an indexed array
        # This uses word splitting to create the array from the string.
        read -r -a LOAD_FROM_PATHS <<< "$CURRENT_PATHS_STRING"

        # Check if the retrieved path list is empty (e.g., if an array was declared but left empty)
        if [ ${#LOAD_FROM_PATHS[@]} -eq 0 ]; then
            echo "--- No LOAD_FROM paths defined for ENCODER: $ENCODER. Skipping." | tee -a "$SAVE_PATH"/terminal.txt
            continue # Skip to the next encoder if no paths are found
        fi

        # Loop through each LOAD_FROM path for the current encoder
        for LOAD_FROM in "${LOAD_FROM_PATHS[@]}"; do
            NOW=$(date +"%Y%m%d_%H%M%S")
            SAVE_PATH="$BASE_SAVE_PATH/$NOW"

            mkdir -p "$SAVE_PATH"

            # Create a copy of the evaluate.py script in the save directory
            cat evaluate.py > "$SAVE_PATH"/evaluate.py

            echo "--- Starting evaluation for LOAD_FROM: $LOAD_FROM ---" | tee -a "$SAVE_PATH"/terminal.txt
            echo "DEBUG: Command being executed:" | tee -a "$SAVE_PATH"/terminal.txt
            echo "python evaluate.py --encoder \"$ENCODER\" --save-path \"$SAVE_PATH\" --load-from \"$LOAD_FROM\" --bs \"$BATCH_SIZE\"" | tee -a "$SAVE_PATH"/terminal.txt


            # Run the evaluation script and redirect output to a log file
            python evaluate.py \
                --encoder "$ENCODER" \
                --save-path "$SAVE_PATH" \
                --load-from "$LOAD_FROM" \
                --max-depth 50 \
                --bs "$BATCH_SIZE" \
                2>&1 | tee -a "$SAVE_PATH"/terminal.txt

            echo "--- Finished evaluation for LOAD_FROM: $LOAD_FROM ---" | tee -a "$SAVE_PATH"/terminal.txt
            echo "" | tee -a "$SAVE_PATH"/terminal.txt # Add a blank line for readability
        done
    else
        echo "--- No specific path string defined for ENCODER: $ENCODER. Skipping." | tee -a "$SAVE_PATH"/terminal.txt
    fi
    echo "=== Finished all evaluations for ENCODER: $ENCODER ===" | tee -a "$SAVE_PATH"/terminal.txt
    echo "" # Add a blank line for readability between encoder runs
done