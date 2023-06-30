#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

python -u occupations_test_construction.py \
  --en_occupations_path "../genbiasmt/occupations/occ.en.txt" \
  --output_path "data/occupations_test.pkl"