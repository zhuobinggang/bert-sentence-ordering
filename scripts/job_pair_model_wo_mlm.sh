#!/bin/bash
#PBS -k doe -j oe
source /work/zb023/research/bert-sentence-ordering/scripts/hpc_common.sh
TQDM_DISABLE=1 python3 train_and_test.py