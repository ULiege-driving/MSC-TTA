#!/bin/bash

SAVE=../pretrained
DATA=../data/
CELL=spatial
SUPERVISION=teacher

cd ./pretraining

python pretraining_static.py --save_folder "$SAVE" --dataset "$DATA" --cell_type "$CELL" --supervision "$SUPERVISION"