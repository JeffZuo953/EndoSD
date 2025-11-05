ENCODER="vits"
BATCH_SIZE=1

# ENCODER="vitb"
# BATCH_SIZE=95

# ENCODER="vitl"
# BATCH_SIZE=40

BASE_SAVE_PATH="/data/eval/dam/rel_inhouse/$ENCODER"

LOAD_FROM="/data/depthanything/depth_anything_v2_${ENCODER}.pth"

NOW=$(date +"%Y%m%d_%H%M%S")

SAVE_PATH=$BASE_SAVE_PATH/$NOW

mkdir -p $SAVE_PATH

cat evaluate_rel.py > $SAVE_PATH/evaluate_rel.py

# Run the evaluation script and redirect output to a log file in the save path
python evaluate_rel.py \
    --encoder $ENCODER \
    --save-path $SAVE_PATH --load-from $LOAD_FROM \
    --bs $BATCH_SIZE \
    --normalize \
    2>&1 | tee -a $SAVE_PATH/terminal.txt

# ENCODER="vits"
# BATCH_SIZE=175

ENCODER="vitb"
BATCH_SIZE=1

# ENCODER="vitl"
# BATCH_SIZE=40

BASE_SAVE_PATH="/data/eval/dam/rel_inhouse/$ENCODER"

LOAD_FROM="/data/depthanything/depth_anything_v2_${ENCODER}.pth"

NOW=$(date +"%Y%m%d_%H%M%S")

SAVE_PATH=$BASE_SAVE_PATH/$NOW

mkdir -p $SAVE_PATH

cat evaluate_rel.py >$SAVE_PATH/evaluate_rel.py

# Run the evaluation script and redirect output to a log file in the save path
python evaluate_rel.py \
    --encoder $ENCODER \
    --save-path $SAVE_PATH --load-from $LOAD_FROM \
    --bs $BATCH_SIZE \
    --normalize \
    2>&1 | tee -a $SAVE_PATH/terminal.txt

# ENCODER="vits"
# BATCH_SIZE=175

# ENCODER="vitb"
# BATCH_SIZE=95

ENCODER="vitl"
BATCH_SIZE=1

BASE_SAVE_PATH="/data/eval/dam/rel_inhouse/$ENCODER"

LOAD_FROM="/data/depthanything/depth_anything_v2_${ENCODER}.pth"

NOW=$(date +"%Y%m%d_%H%M%S")

SAVE_PATH=$BASE_SAVE_PATH/$NOW

mkdir -p $SAVE_PATH

cat evaluate_rel.py >$SAVE_PATH/evaluate_rel.py

# Run the evaluation script and redirect output to a log file in the save path
python evaluate_rel.py \
    --encoder $ENCODER \
    --save-path $SAVE_PATH --load-from $LOAD_FROM \
    --bs $BATCH_SIZE \
    --normalize \
    2>&1 | tee -a $SAVE_PATH/terminal.txt
