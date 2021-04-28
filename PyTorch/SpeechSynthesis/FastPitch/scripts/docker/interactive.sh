#!/usr/bin/env bash

PORT=${PORT:-8888}
# GPU="1,3"
    # --gpus '"device=$GPU"' \
docker run -it \
    --rm \
    --gpus '"device=1,3"' \
    --ipc=host \
    -e CUDA_VISIBLE_DEVICES \
    -p $PORT:$PORT \
    -v $PWD:/workspace/fastpitch/ fastpitch:latest bash 

    # --cpuset-cpus="12-23" \