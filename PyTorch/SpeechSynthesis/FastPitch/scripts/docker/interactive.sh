#!/usr/bin/env bash

PORT=${PORT:-8889}
# GPU="1,3"
    # --gpus '"device=$GPU"' \
    # --user $(id -u):$(id -g) \
    # --gpus '"device=0"' \
    # -e CUDA_VISIBLE_DEVICES \
docker run -it \
    --rm \
    --ipc=host \
    -p $PORT:$PORT \
    -v $PWD:/workspace/fastpitch/ fastpitch:latest bash 

    # --cpuset-cpus="12-23" \

# Not sure why, but I can't add `--user $(id -u):$(id -g)`; 
# this results in a strange error when I do 
# `from inference import load_and_setup_model`
