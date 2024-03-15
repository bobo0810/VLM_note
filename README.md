# VLM_note
大模型学习笔记

[TOC]



# Tips

1. 问题：一阶段的数据量小、batch大，导致模型回答重复、胡言乱语。 如何解决？

    解决：多模态大模型欠拟合，调小Batch。

2. 问题：一阶段训练的模型回答重复、大量无意义符号。如何解决？

   解决：二阶段指令微调后大幅改善。



## 基本概念&术语

1. SYSTEM

   定义模型的角色

   ```python
   SYSTEM=('A chat between a curious user and an artificial '
   'intelligence assistant. The assistant gives '
   'helpful, detailed, and polite answers to the '
   'user's questions. {system}\n '),
   ```

2. INSTRUCTION  

   用户输入和助手回答的格式

   ```python
   'USER: {input} ASSISTANT:'
   ```

3. SEP  

   “分隔符”（Separator）的缩写，表示不同对话部分的分隔符

4. prompt_template

   提示词模板

   ```python
   dict(
     SYSTEM=('A chat between a curious user and an artificial '
             'intelligence assistant. The assistant gives '
             'helpful, detailed, and polite answers to the '
             'user\'s questions. {system}\n '),
     INSTRUCTION=('USER: {input} ASSISTANT:'),
     SEP='\n'),
   ```

   

# Code

## 1. Tokenizer文本编解码

```python

# 导入所需库
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/xxx/lmsys/vicuna-7b-v1.5")

# 示例文本
text = "这是一个示例文本。"

# 对文本进行编码,生成token_id
encoded_text = tokenizer.encode(text)

# 将编码后的序列id解码为原始文本
decoded_text = tokenizer.decode(encoded_text)

print("原始文本：", text)
print("解码后的文本：", decoded_text)
```



## 
