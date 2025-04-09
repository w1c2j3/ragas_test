# RAG测试数据集调研

## 公开可用的RAG测试数据集

### 1. MTEB（Massive Text Embedding Benchmark）
- **来源**: [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)
- **特点**: 包含8种语言、58个数据集、的文本嵌入基准测试集合
- **适用场景**: 评估文本嵌入模型和检索系统性能
- **使用方法**: 可以使用其中的检索任务相关数据集进行RAG评估

### 2. KILT（Knowledge-Intensive Language Tasks）
- **来源**: [KILT GitHub](https://github.com/facebookresearch/KILT)
- **特点**: 基于维基百科的知识密集型语言任务集合
- **适用场景**: 包含实体链接、开放域QA、对话等任务
- **数据格式**: 每个任务包含查询和与维基百科相关的证据文档

### 3. MS MARCO
- **来源**: [MS MARCO](https://microsoft.github.io/msmarco/)
- **特点**: 大规模信息检索数据集，包含真实搜索查询
- **适用场景**: 评估文档检索和问答系统
- **优势**: 数据量大，涵盖多种查询类型

### 4. NQ（Natural Questions）
- **来源**: [Natural Questions](https://ai.google.com/research/NaturalQuestions)
- **特点**: 基于Google搜索查询的真实问题
- **适用场景**: 开放域问答和文档检索评估
- **格式**: 包含问题、长答案(段落)和短答案

### 5. HotpotQA
- **来源**: [HotpotQA](https://hotpotqa.github.io/)
- **特点**: 多跳推理问答数据集
- **适用场景**: 评估RAG系统的多文档推理能力
- **特色**: 需要从多个文档中收集信息来回答问题

## 中文RAG测试数据集

### 1. C-MTEB
- **来源**: [C-MTEB GitHub](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
- **特点**: 中文大规模文本嵌入基准测试
- **适用场景**: 中文文本检索和语义匹配评估
- **包含任务**: 检索、分类、聚类、重排序等

### 2. DuReader
- **来源**: [DuReader](https://github.com/baidu/DuReader)
- **特点**: 百度发布的中文阅读理解数据集
- **适用场景**: 中文问答系统评估
- **数据来源**: 基于真实搜索日志和网页内容

### 3. CMRC 2018
- **来源**: [CMRC 2018](https://github.com/ymcui/cmrc2018)
- **特点**: 中文机器阅读理解数据集
- **适用场景**: 评估模型从给定文档中抽取答案的能力
- **特色**: 人工标注的高质量问题和答案

## 使用ragas库生成测试数据集

除了使用现有数据集，ragas库还提供了自动生成测试数据的功能：

```python
from ragas.testset import TestsetGenerator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

# 加载文档
loader = TextLoader("path/to/your/document.txt")
documents = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# 创建测试集生成器
generator = TestsetGenerator.from_langchain(
    vectorstore=vectorstore,
    llm=ChatOpenAI(temperature=0.2)
)

# 生成测试集
testset = generator.generate(num_questions=10)

# 保存测试集
testset.to_pandas().to_csv("testset.csv")
```

## 数据集选择建议

1. **初始测试阶段**: 使用ragas自动生成小型测试集，针对自己的文档进行定制化测试
2. **验证系统性能**: 使用公开基准数据集(如KILT或MS MARCO)进行评估，便于与其他系统比较
3. **中文场景评估**: 优先选择C-MTEB或DuReader等中文数据集
4. **多跳推理能力评估**: 使用HotpotQA测试系统处理复杂问题的能力 