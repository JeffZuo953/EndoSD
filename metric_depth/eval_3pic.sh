declare -A BATCH_SIZES
BATCH_SIZES["vits"]=1

declare -A LOAD_FROM_PATH_STRINGS

LOAD_FROM_PATH_STRINGS["vits"]="/data/train_combined/best_abs_rel.pth \
/data/train_combined/latest.pth \
/data/train_combined/checkpoint_epoch_0.pth \
/data/train_combined/checkpoint_epoch_10.pth \
/data/train_combined/checkpoint_epoch_20.pth \
/data/train_combined/checkpoint_epoch_30.pth \
/data/train_combined/checkpoint_epoch_40.pth \
/data/train_combined/checkpoint_epoch_50.pth \
/data/train_combined/checkpoint_epoch_60.pth \
/data/train_combined/checkpoint_epoch_70.pth \
/data/train_combined/checkpoint_epoch_80.pth \
/data/train_combined/checkpoint_epoch_90.pth \
/data/train_combined/checkpoint_epoch_100.pth \
/data/train_combined/checkpoint_epoch_110.pth"

for ENCODER in "${!BATCH_SIZES[@]}"; do
    BATCH_SIZE="${BATCH_SIZES[$ENCODER]}"
    BASE_SAVE_PATH="/data/debug/dam/inhouse/$ENCODER"

    echo "=== Starting evaluations for ENCODER: $ENCODER with BATCH_SIZE: $BATCH_SIZE ==="

    if [[ -v LOAD_FROM_PATH_STRINGS[$ENCODER] ]]; then
        CURRENT_PATHS_STRING="${LOAD_FROM_PATH_STRINGS[$ENCODER]}"

        read -r -a LOAD_FROM_PATHS <<< "$CURRENT_PATHS_STRING"

        if [ ${#LOAD_FROM_PATHS[@]} -eq 0 ]; then
            echo "--- No LOAD_FROM paths defined for ENCODER: $ENCODER. Skipping." | tee -a "$SAVE_PATH"/terminal.txt
            continue
        fi

        for LOAD_FROM in "${LOAD_FROM_PATHS[@]}"; do
            FILENAME=$(basename "$LOAD_FROM")
            DIRNAME=$(basename "$(dirname "$LOAD_FROM")")

            if [[ "$DIRNAME" == "." ]]; then
                MODEL_SUFFIX="${FILENAME%.*}"
            else
                MODEL_SUFFIX="${DIRNAME}_${FILENAME%.*}"
            fi

            SAVE_PATH="$BASE_SAVE_PATH/$MODEL_SUFFIX"

            mkdir -p "$SAVE_PATH"

            cat evaluate.py > "$SAVE_PATH"/evaluate.py

            echo "--- Starting evaluation for LOAD_FROM: $LOAD_FROM ---" | tee -a "$SAVE_PATH"/terminal.txt
            echo "DEBUG: Command being executed:" | tee -a "$SAVE_PATH"/terminal.txt
            echo "python evaluate.py --encoder \"$ENCODER\" --save-path \"$SAVE_PATH\" --load-from \"$LOAD_FROM\" --bs \"$BATCH_SIZE\"" | tee -a "$SAVE_PATH"/terminal.txt

            python evaluate.py \
                --encoder "$ENCODER" \
                --save-path "$SAVE_PATH" \
                --load-from "$LOAD_FROM" \
                --max-depth 50 \
                --bs "$BATCH_SIZE" \
                2>&1 | tee -a "$SAVE_PATH"/terminal.txt

            echo "--- Finished evaluation for LOAD_FROM: $LOAD_FROM ---" | tee -a "$SAVE_PATH"/terminal.txt
            echo "" | tee -a "$SAVE_PATH"/terminal.txt
        done
    else
        echo "--- No specific path string defined for ENCODER: $ENCODER. Skipping." | tee -a "$SAVE_PATH"/terminal.txt
    fi
    echo "=== Finished all evaluations for ENCODER: $ENCODER ===" | tee -a "$SAVE_PATH"/terminal.txt
    echo ""
done
