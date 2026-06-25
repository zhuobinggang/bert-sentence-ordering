# 考察n pass的信息
from n_pass_with_critic import *
import common

def run():
    ckps = common.search_files_in_directory('_vanilla_sind_', directory="./checkpoints")
    bert = default_trained_bert(ckps[0])
    