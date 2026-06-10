#!/bin/bash
#PBS -k doe -j oe
source /work/zb023/research/bert-sentence-ordering/scripts/hpc_common.sh
python3 run_train.py