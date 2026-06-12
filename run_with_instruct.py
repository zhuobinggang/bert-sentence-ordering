from sind import *
import common

assert common.args.instruction, "Please use the -inst flag to enable instruction in the input"

default_bert_decode_acc('test')