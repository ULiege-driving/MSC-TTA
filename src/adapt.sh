#!/bin/bash

PRETRAINED_FOLDER=../pretrained
DATA=../data/
SAVE=../confusion_matrices
CELL=spatial
SUPERVISION=carla
PRETRAINING_TYPE=scratch

cd ./online

python adapt_static.py --dataset "$DATA" --save_folder "$SAVE" --pretraining_path "$PRETRAINED_FOLDER" --pretraining "$PRETRAINING_TYPE" --cell_type "$CELL" --supervision "$SUPERVISION"



