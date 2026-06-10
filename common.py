import re
import logging
import os
from pathlib import Path
import argparse

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