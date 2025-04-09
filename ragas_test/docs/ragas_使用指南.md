# Ragas使用指南

## 安装

首先需要安装ragas库：

```bash
pip install ragas
```

如果需要与其他框架集成，可能需要安装额外的依赖：

```bash
pip install ragas[langchain]  # 与LangChain集成
```

## 基本评估流程

Ragas主要用于评估RAG（检索增强生成）系统的质量，评估流程大致如下：

### 1. 准备评估数据

评估数据通常包含以下组件：
- 问题（query）
- 上下文文档（contexts）
- 模型生成的回答（response）
- 参考答案（ground_truth）- 可选

示例：

```python
from ragas import EvaluationDataset

# 准备评估数据
data = [
    {
        "question": "什么是RAG系统?",
        "contexts": ["RAG代表检索增强生成，是一种将检索系统与生成模型结合的方法...", "..."],
        "answer": "RAG是'检索增强生成'的缩写，它结合了检索系统和生成模型，通过先检索相关文档然后基于这些文档生成回答。",
        "ground_truth": "RAG是检索增强生成技术，结合检索和生成能力来提高模型回答的准确性。" # 可选
    },
    # 更多评估示例...
]

# 创建评估数据集
eval_dataset = EvaluationDataset(data)
```

### 2. 选择评估指标

Ragas提供多种评估指标：

```python
from ragas.metrics import (
    faithfulness,    # 忠实度：回答是否忠实于上下文
    answer_relevancy,  # 回答相关性：回答是否与问题相关
    context_relevancy,  # 上下文相关性：上下文是否与问题相关
    context_recall,  # 上下文召回率：上下文是否包含回答所需信息
)
```

### 3. 运行评估

```python
from ragas import evaluate

# 运行评估
result = evaluate(
    eval_dataset, 
    metrics=[faithfulness, answer_relevancy, context_relevancy, context_recall]
)

# 查看评估结果
print(result)
```

## 生成测试数据

Ragas还提供了测试数据生成功能：

```python
from ragas.testset import TestsetGenerator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# 准备文档
documents = [...]  # 文档列表

# 创建向量存储
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# 初始化测试集生成器
generator = TestsetGenerator.from_langchain(
    vectorstore=vectorstore,
    llm=ChatOpenAI(temperature=0.3)
)

# 生成测试集
testset = generator.generate(num_questions=10)
```

## 常见问题与解决方案

1. **评估结果中出现NaN值**
   - 可能原因：模型输出不是JSON可解析的，或者某些评分案例不理想
   - 解决方案：检查模型输出格式，确保所有字段都正确填充

2. **如何使评估结果更加可解释**
   - 最佳方式是追踪和记录评估，然后使用LLM跟踪检查结果

3. **选择最佳的开源模型**
   - 没有唯一正确的答案，最佳模型取决于GPU容量和处理的数据类型
   - 建议探索较新的、被广泛接受的具有强大通用能力的模型 