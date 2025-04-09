"""
使用ragas生成RAG测试数据集的示例
"""

import os
from dotenv import load_dotenv
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# 加载环境变量
load_dotenv()

def generate_test_dataset(
    document_path="./data/sample_document.txt",
    output_path="./data/generated_testset.csv",
    num_questions=5
):
    """
    使用ragas库生成RAG测试数据集
    
    Args:
        document_path: 文档路径
        output_path: 输出测试集路径
        num_questions: 生成问题数量
    """
    print(f"开始从文档 {document_path} 生成测试数据集...")
    
    try:
        # 加载文档
        loader = TextLoader(document_path)
        documents = loader.load()
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"文档已分割为 {len(splits)} 个片段")
        
        # 创建向量存储
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # 创建LLM模型
        llm = ChatOpenAI(temperature=0.2)
        
        # 初始化测试集生成器
        generator = TestsetGenerator.from_langchain(
            vectorstore=vectorstore,
            llm=llm,
            # 使用不同类型的问题生成演变
            evolutions=[
                simple(max_questions=num_questions // 3 + 1),  # 简单问题
                reasoning(max_questions=num_questions // 3 + 1),  # 需要推理的问题
                multi_context(max_questions=num_questions // 3 + 1)  # 需要多个上下文的问题
            ]
        )
        
        # 生成测试集
        print("正在生成测试集，这可能需要几分钟...")
        testset = generator.generate()
        
        # 转换为pandas DataFrame并保存
        testset_df = testset.to_pandas()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存到CSV
        testset_df.to_csv(output_path, index=False)
        print(f"测试集已生成并保存到 {output_path}")
        
        # 显示测试集统计信息
        print(f"\n生成的测试集包含 {len(testset_df)} 个问题")
        print(f"问题类型分布: {testset_df['type'].value_counts().to_dict()}")
        
        # 显示几个示例问题
        print("\n示例问题:")
        for i, row in testset_df.head(3).iterrows():
            print(f"问题 {i+1}: {row['question']}")
            print(f"类型: {row['type']}")
            print(f"回答: {row['ground_truth']}")
            print("-" * 50)
        
        return testset_df
        
    except Exception as e:
        print(f"生成测试集时出错: {e}")
        print("注意: 这个脚本需要OpenAI API密钥才能运行")
        return None

if __name__ == "__main__":
    # 创建一个简单的示例文档用于演示
    sample_text = """
    人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    这些任务包括视觉感知、语音识别、决策制定和语言翻译等。
    
    机器学习是AI的一个子领域，它使用统计技术使计算机系统能够从数据中"学习"，而无需明确编程。
    深度学习是机器学习的一个特殊分支，它使用神经网络来模拟人脑的工作方式。
    
    自然语言处理(NLP)是AI的另一个关键领域，专注于使计算机能够理解、解释和生成人类语言。
    NLP技术被用于各种应用，如虚拟助手、语言翻译和情感分析。
    
    计算机视觉是AI的一个领域，它训练计算机"看到"和理解视觉世界的内容。
    这包括图像识别、物体检测和场景重建等任务。
    
    强化学习是一种机器学习方法，其中代理通过与环境交互并接收奖励或惩罚来学习如何行动。
    这种方法已被用于训练AI系统玩游戏、控制机器人和优化系统等。
    """
    
    # 确保数据目录存在
    os.makedirs("./data", exist_ok=True)
    
    # 创建示例文档
    with open("./data/sample_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # 生成测试集
    generate_test_dataset() 