#!/usr/bin/env bash

PORT=${PORT:-8888}
GPU=1
docker run -it \
    --gpus device=$GPU \
    --cpuset-cpus="0-4" \
    --rm \
    --ipc=host \
    -e CUDA_VISIBLE_DEVICES \
    -v $PWD:/workspace/fastpitch/ fastpitch:latest bash 

    # -p $PORT:$PORT \