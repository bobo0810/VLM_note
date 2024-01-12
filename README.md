# VLM_note
大模型学习笔记

[TOC]





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

