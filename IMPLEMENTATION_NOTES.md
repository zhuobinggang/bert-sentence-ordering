# Multi-Step Decoding Implementation

## Overview
实现了多步解码逻辑，在SIND数据集上进行句子排序任务。

## 核心算法

### `multi_step_decode()` 函数
**目的**: 迭代式预测，每次选择置信度最高的位置

**算法流程**:
1. 初始化输入张量（包含5个MASK令牌）
2. **循环**直到没有MASK令牌：
   - 获取BERT模型对所有MASK位置的logits输出 [num_masks, 30522]
   - 提取每个MASK位置的最大logit值（置信度）
   - **选择最高置信度的位置** `best_position`
   - 获取该位置的预测token ID
   - 将token ID转换为标签（1-5）
   - 用预测结果**填充**该位置的MASK令牌
3. 返回预测标签列表

### 关键设计点

1. **置信度度量**: 使用logits的最大值作为置信度
   ```python
   max_logits_per_position = mask_logits.max(dim=-1).values  # [num_masks]
   best_mask_idx = max_logits_per_position.argmax()  # 选最高的
   ```

2. **动态填充**: 每次预测后，用预测的token ID替换MASK令牌
   ```python
   input_ids_tensor[0, best_position] = predicted_token_id
   ```

3. **标签转换**: 使用 `reversed_dict` 将token ID映射到序号标签
   ```python
   predicted_label = reversed_dict.get(predicted_token_id, 5)
   ```

## 使用方法

### 单样本解码
```python
from multi_step_decode import multi_step_decode
from reader import default_bert, load_checkpoint

bert = default_bert()
load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')

# 获取模型预测
predictions = multi_step_decode(input_ids, attention_mask, bert)
```

### 批量验证
```python
from multi_step_decode import valid_bert_multi_step

result = valid_bert_multi_step(bert, split='test')
# 返回: TestResult(tau=..., acc=..., pmr=...)
```

## 与单步解码的对比

### 单步解码 (`reader.py` - `valid_bert_batched`)
- 一次性预测所有5个MASK位置
- 每个位置独立预测，无相互影响
- 可能产生重复标签

### 多步解码 (`multi_step_decode.py`)
- 每次选择最高置信度的位置预测
- 后续预测考虑前面的填充结果
- 更接近自回归模型的工作方式
- 可能产生不同的结果（通常更好）

## 实现细节

- **device支持**: 自动检测CUDA/CPU
- **tensor操作**: 使用torch张量进行高效计算
- **无梯度**: 使用`torch.no_grad()`确保推理模式
- **错误处理**: 含有断言确保输出维度正确

## 调用示例

在 `__main__` 块中提供了完整使用示例：
```python
if __name__ == '__main__':
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')
    result = valid_bert_multi_step(bert, 'test')
```
