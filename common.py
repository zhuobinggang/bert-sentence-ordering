import re
import logging
import os
from pathlib import Path
import argparse
from functools import lru_cache

LOCAL = False
dir_path = Path("/work/zb023")
if dir_path.is_dir():
    LOCAL = False
    print("处于HPC环境")
    dataset_base = '/work/zb023/datasets'
else:
    LOCAL = True
    print("处于本地测试环境")
    dataset_base = '/home/zhuobinggang/research/datasets'


parser = argparse.ArgumentParser()
parser.add_argument('-inst', '--instruction', action='store_true', help='Whether to use instruction in the input')
# parser.add_argument('-nav', '--navigator', action='store_true', help='Whether to use the navigator')
# parser.add_argument('-cgen', '--cmdgen', action='store_true', help='Whether to use command generate game and dataset')
# parser.add_argument('-rawcmd', '--rawcmd', action='store_true', help='Whether to use raw commands without filtering')
args = parser.parse_args()


@lru_cache(maxsize=1)
def print_once(msg):
    print(msg)