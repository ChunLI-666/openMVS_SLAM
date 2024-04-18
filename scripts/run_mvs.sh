#!/bin/bash

# 设置工作目录
WORK_DIR="/home/charles/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/MVS_Workspace"
CMD_DIR="/home/charles/projects/openMVS_build/bin"
# 输入和输出文件
INPUT_MVS_FRAME="$WORK_DIR/mvs_frame_result.txt"
DENSIFY_OUTPUT="mvs_densify.mvs"
RECONSTRUCT_OUTPUT="mvs_reconstruct.mvs"
REFINE_OUTPUT="mvs_refine.mvs"
TEXTURE_OUTPUT="mvs_texture.mvs"

# DensifyPointCloud
echo "Running DensifyPointCloud..."
$CMD_DIR/DensifyPointCloud -w "$WORK_DIR" -i "$INPUT_MVS_FRAME" -o "$WORK_DIR/$DENSIFY_OUTPUT"

# ReconstructMesh
echo "Running ReconstructMesh..."
$CMD_DIR/ReconstructMesh -w "$WORK_DIR" -i "$WORK_DIR/$DENSIFY_OUTPUT" -o "$WORK_DIR/$RECONSTRUCT_OUTPUT"

# RefineMesh
echo "Running RefineMesh..."
$CMD_DIR/RefineMesh -w "$WORK_DIR" -i "$WORK_DIR/$RECONSTRUCT_OUTPUT" -o "$WORK_DIR/$REFINE_OUTPUT"

# TextureMesh
echo "Running TextureMesh..."
$CMD_DIR/TextureMesh -w "$WORK_DIR" -i "$WORK_DIR/$REFINE_OUTPUT" -o "$WORK_DIR/$TEXTURE_OUTPUT"

echo "All commands executed successfully."
