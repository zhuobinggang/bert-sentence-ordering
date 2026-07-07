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
parser.add_argument('-n', '--nsteps', type=int, default=1, help='Number of steps for decoding')
parser.add_argument('-rp', '--repeats', type=int, default=1, help='Number of repeats for training')
parser.add_argument('-off', '--offset', type=int, default=1, help='Offset suffix for training repeats')
args = parser.parse_args()

def get_time_str():
    from datetime import datetime
    dt = datetime.now()
    # %f 直接返回 6 位微秒数
    return dt.strftime('%Y%m%d_%H%M%S_%f')

LOG_FILE = f'log/log_{get_time_str()}.log'

DEBUG = False
if DEBUG:
    logging.basicConfig(filename=LOG_FILE, filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filename=LOG_FILE, filemode='w', level=logging.WARNING)

@lru_cache(maxsize=99) # 记录100个
def print_once(msg):
    print(msg)


def get_writer(base_log_dir="runs"):
    import time
    from tensorboardX import SummaryWriter
    # 使用纳秒级时间戳
    #timestamp = time.time_ns()
    log_dir = os.path.join(base_log_dir, f'run_{get_time_str()}')
    writer = SummaryWriter(log_dir=log_dir)
    writer.global_step = 0
    return writer

def cal_mean_std(scores):
    import numpy as np
    std_np = np.std(scores, ddof=1)
    mean_np = np.mean(scores)
    print(f"Numpy: {mean_np:.4f} ± {std_np:.4f}")
    return f'{mean_np:.4f} ± {std_np:.4f}'

def list_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if a != b:
            return False
    return True

def list_in(a, container):
    for b in container:
        if list_equal(a, b):
            return True
    return False
    
# use str(matching_file) to get the path string
def search_files_in_directory(search_string, directory="./checkpoints"):
    from pathlib import Path
    directory_path = Path(directory)
    matching_file_abs_paths = [str(file) for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    return matching_file_abs_paths

def print_only_once(msg):
    if not hasattr(print_only_once, "has_printed"):
        print(msg)
        print_only_once.has_printed = True


def resort_paragraph(paragraph, predicted_label):
    """ 根据predicted_label重排paragraph，得到新的段落顺序 """
    ordered_paragraph = [None] * 5
    for idx, label in enumerate(predicted_label):
        ordered_paragraph[label - 1] = paragraph[idx] # label是1-5的索引
    return ordered_paragraph


# 将排好的段落s1, s2, s3, s4, s5按predicted_label重建未排序的段落
def recover_unsorted_paragraph(paragraph, predicted_label):
    """ 根据predicted_label重排paragraph，得到新的段落顺序 """
    unsorted_paragraph = [None] * 5
    for idx, label in enumerate(predicted_label):
        unsorted_paragraph[idx] = paragraph[label - 1] # label是1-5的索引
    return unsorted_paragraph


def add_one(lst):
    return [x + 1 for x in lst]