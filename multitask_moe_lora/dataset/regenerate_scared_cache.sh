#!/bin/bash
# Complete workflow to regenerate SCARED dataset cache with correct depth scaling

set -e  # Exit on error

CACHE_DIR="/data/ziyi/multitask/data/SCARED/cache"
CODE_DIR="/data/ziyi/multitask/code/DepthAnythingV2/multitask_moe_lora"

echo "================================================================================"
echo "SCARED Dataset Cache Regeneration Workflow"
echo "================================================================================"
echo ""

# Step 1: Dry run to show what will be deleted
echo "Step 1: Preview files to be deleted..."
echo "--------------------------------------------------------------------------------"
python $CODE_DIR/dataset/regenerate_cache.py "$CACHE_DIR" --dry-run
echo ""

# Ask for confirmation to proceed
read -p "Do you want to proceed with cleaning and regeneration? [y/N]: " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 2: Clean old cache
echo ""
echo "Step 2: Cleaning old cache files..."
echo "--------------------------------------------------------------------------------"
python $CODE_DIR/dataset/regenerate_cache.py "$CACHE_DIR"
echo ""

# Step 3: Regenerate cache
echo ""
echo "Step 3: Regenerating cache with correct depth scaling..."
echo "--------------------------------------------------------------------------------"
cd "$CODE_DIR"
python -m dataset.cache_utils_data
echo ""

# Step 4: Analyze new cache
echo ""
echo "Step 4: Analyzing new cache statistics..."
echo "--------------------------------------------------------------------------------"
python -m dataset.analyze_cached_depth "$CACHE_DIR/train_cache.txt"
echo ""

echo "================================================================================"
echo "Cache regeneration complete!"
echo "================================================================================"
echo ""
echo "Please verify that the depth statistics look correct:"
echo "  - Depth range should be approximately [0.037, 0.272]"
echo "  - Mean should be around 0.12-0.13"
echo "  - All values should NOT be 0.3"
echo ""
echo "Report files:"
echo "  - $CACHE_DIR/depth_statistics_report.txt"
echo "  - $CACHE_DIR/depth_statistics_report.json"
echo "  - $CACHE_DIR/train_cache_depth_analysis.txt"
echo "  - $CACHE_DIR/train_cache_depth_analysis.json"
echo ""
