# 多步解码逻辑：
# 每次预测之后选择置信度最高的一个位置进行输出，填充到输入中，继续下一轮预测，直到所有位置都被预测出来。

from reader import *

bert = default_bert()
load_checkpoint(bert, './checkpoints/SIND_best_e0.pth' )
