# RAG评估工具包

一个完整的RAG（检索增强生成）系统评估工具包，支持数据集处理、评估执行和结果可视化。

## 功能特点

- **全面的评估指标支持**：支持Context Precision、Answer Relevancy、Faithfulness、Context Recall等多种RAG评估指标
- **灵活的数据集处理**：支持数据集下载、处理和自定义测试集生成
- **自定义API支持**：支持配置和使用自定义API进行RAG评估
- **直观的可视化界面**：HTML格式的评估结果展示，包含指标摘要和详细数据
- **完整的评估流程**：从数据集准备到评估执行再到结果分析的全流程支持

## 安装

```bash
git clone https://github.com/yourusername/rag-evaluation-tool.git
cd rag-evaluation-tool
pip install -r requirements.txt
```

## 依赖

- ragas>=0.2.14
- langchain>=0.1.0
- langchain-openai>=0.0.2
- faiss-cpu>=1.7.4
- openai>=1.2.0
- datasets>=2.0.0
- pandas>=1.3.0
- numpy>=1.20.0
- python-dotenv>=1.0.0
- transformers>=4.33.0
- tiktoken>=0.5.1
- argparse>=1.4.0

## 项目结构

```
rag-evaluation-tool/
│
├── src/                          # 源代码
│   ├── __init__.py               # 包初始化
│   ├── ragas_example.py          # RAG评估示例
│   ├── custom_api_client.py      # 自定义API客户端
│   ├── test_api.py               # API测试工具
│   ├── evaluate_rag.py           # RAG评估核心功能
│   ├── check_dataset.py          # 数据集检查工具
│   ├── download_dataset.py       # 数据集下载工具
│   ├── custom_testset_generator.py # 自定义测试集生成器
│   ├── testset_generator_example.py # 测试集生成示例
│   ├── data_processor.py         # 数据处理模块
│   └── visualize.py              # 可视化模块
│
├── data/                         # 数据集目录
│   └── msmarco_for_ragas/        # MSMARCO数据集
│
├── evaluation_results/           # 评估结果文件
│   └── *.txt                     # 评估结果文件
│
├── docs/                         # 文档目录
│
├── .env                          # 环境变量配置
├── .env.example                  # 环境变量示例
├── requirements.txt              # 依赖包列表
└── README.md                     # 项目说明
```

## 使用方法

### 1. 环境配置

复制 `.env.example` 到 `.env` 并配置必要的环境变量：

```bash
cp .env.example .env
```

### 2. API测试

测试API连接是否正常：

```bash
python -m src.test_api
```

### 3. 数据集准备

#### 下载数据集

```bash
python -m src.download_dataset --dataset msmarco --output_dir data/msmarco_for_ragas
```

#### 生成自定义测试集

```bash
python -m src.testset_generator_example
```

### 4. 执行RAG评估

使用示例脚本进行RAG评估：

```bash
python -m src.ragas_example
```

### 5. 可视化评估结果

```bash
python -m src.visualize evaluation_results/rag_eval_*.txt
```

## 主要功能模块

### 1. 自定义API客户端 (custom_api_client.py)

- 支持自定义API配置
- 支持流式和非流式响应
- 与LangChain框架集成

### 2. 数据集处理 (data_processor.py)

- 数据集加载和预处理
- 字段映射和转换
- 数据格式验证

### 3. 评估执行 (evaluate_rag.py)

- 多指标评估支持
- 批量评估处理
- 结果统计和分析

### 4. 结果可视化 (visualize.py)

- HTML格式结果展示
- 交互式文本查看
- 指标统计图表

## 评估指标说明

该工具支持以下RAG评估指标：

1. **Context Precision (上下文精度)**：衡量检索到的上下文与问题的相关性
2. **Answer Relevancy (答案相关性)**：衡量生成的答案与问题的相关性
3. **Faithfulness (忠实性)**：衡量生成的答案对检索上下文的忠实程度
4. **Context Recall (上下文召回率)**：衡量检索系统召回所有相关信息的能力
5. **Context Entities Recall (上下文实体召回率)**：衡量检索系统召回关键实体的能力
6. **Answer Correctness (答案正确性)**：衡量生成答案的正确性
7. **Answer Similarity (答案相似度)**：衡量生成答案与参考答案的相似程度

## 参考资源

- [Ragas: Evaluation of Retrieval Augmented Generation](https://docs.ragas.io/)
- [LangChain RAG Evaluation](https://python.langchain.com/docs/guides/evaluation/)

## 贡献

欢迎提交Pull Request或Issue！

## 许可

MIT License 
