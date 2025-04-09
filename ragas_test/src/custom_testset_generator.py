"""
使用自定义API生成RAG测试数据
"""

import os
import random
from typing import List, Dict, Any, Optional
import pandas as pd
from src.custom_api_client import CustomLLMClient

class SimpleTestsetGenerator:
    """一个简化版的测试集生成器，使用自定义API"""
    
    def __init__(self, documents: List[str], api_url=None, api_key=None, model_name=None):
        """初始化测试集生成器
        
        Args:
            documents: 用于生成测试的文档列表
            api_url: 自定义API URL
            api_key: 自定义API密钥
            model_name: 模型名称
        """
        self.documents = documents
        self.llm = CustomLLMClient()
        
        if api_url:
            self.llm.api_url = api_url
        if api_key:
            self.llm.api_key = api_key
        if model_name:
            self.llm.model_name = model_name
            
    def generate_question(self, context: str) -> str:
        """从给定的上下文生成问题
        
        Args:
            context: 文档内容
            
        Returns:
            生成的问题
        """
        prompt = f"""
根据以下文档内容，生成一个有意义的问题：

文档内容:
"{context}"

生成一个问题，问题应该:
1. 直接与文档内容相关
2. 可以通过文档内容回答
3. 包含特定细节而非泛泛而谈
4. 不要在问题中包含对答案的明显提示

仅返回生成的问题，不需要任何额外解释。
"""
        try:
            question = self.llm._call(prompt)
            # 清理响应，确保只返回问题
            question = question.strip().strip('"').strip("'")
            return question
        except Exception as e:
            print(f"生成问题时出错: {e}")
            return "这个文档在讨论什么？"
    
    def generate_answer(self, context: str, question: str) -> str:
        """根据上下文和问题生成答案
        
        Args:
            context: 文档内容
            question: 问题
            
        Returns:
            生成的答案
        """
        prompt = f"""
根据以下文档内容，回答给定的问题：

文档内容:
"{context}"

问题:
"{question}"

仅提供答案，不要包含任何额外解释。只根据提供的文档内容回答，不要添加额外信息。
"""
        try:
            answer = self.llm._call(prompt)
            return answer.strip()
        except Exception as e:
            print(f"生成答案时出错: {e}")
            return "无法从文档中获取答案。"
    
    def generate_testset(self, num_questions: int = 5) -> pd.DataFrame:
        """生成测试集
        
        Args:
            num_questions: 要生成的问题数量
            
        Returns:
            包含问题、上下文和答案的DataFrame
        """
        # 确保不超过文档数量
        num_questions = min(num_questions, len(self.documents))
        
        # 随机选择文档
        selected_docs = random.sample(self.documents, num_questions)
        
        testset = []
        
        print(f"开始生成{num_questions}个测试问题...")
        
        for i, doc in enumerate(selected_docs):
            print(f"生成问题 {i+1}/{num_questions}...")
            
            # 生成问题
            question = self.generate_question(doc)
            
            # 生成答案
            print(f"为问题 {i+1} 生成答案...")
            answer = self.generate_answer(doc, question)
            
            testset.append({
                "question": question,
                "contexts": [doc],
                "answer": answer,
                "context_index": i
            })
            
            print(f"完成问题 {i+1}!")
            
        return pd.DataFrame(testset)
    
    def save_testset(self, testset: pd.DataFrame, output_path: str):
        """保存测试集到CSV文件
        
        Args:
            testset: 测试集DataFrame
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存到CSV
        testset.to_csv(output_path, index=False)
        print(f"测试集已保存到 {output_path}")


# 使用示例
if __name__ == "__main__":
    # 示例文档
    sample_docs = [
        "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括视觉感知、语音识别、决策制定和语言翻译等。",
        "机器学习是AI的一个子领域，它使用统计技术使计算机系统能够从数据中"学习"，而无需明确编程。深度学习是机器学习的一个特殊分支，它使用神经网络来模拟人脑的工作方式。",
        "自然语言处理(NLP)是AI的另一个关键领域，专注于使计算机能够理解、解释和生成人类语言。NLP技术被用于各种应用，如虚拟助手、语言翻译和情感分析。",
        "计算机视觉是AI的一个领域，它训练计算机"看到"和理解视觉世界的内容。这包括图像识别、物体检测和场景重建等任务。",
        "强化学习是一种机器学习方法，其中代理通过与环境交互并接收奖励或惩罚来学习如何行动。这种方法已被用于训练AI系统玩游戏、控制机器人和优化系统等。"
    ]
    
    # 创建测试集生成器
    generator = SimpleTestsetGenerator(documents=sample_docs)
    
    # 生成测试集
    testset = generator.generate_testset(num_questions=3)
    
    # 显示生成的测试集
    print("\n生成的测试集:")
    for i, row in testset.iterrows():
        print(f"\n问题 {i+1}: {row['question']}")
        print(f"上下文: {row['contexts'][0][:100]}...")
        print(f"答案: {row['answer']}")
    
    # 保存测试集
    generator.save_testset(testset, "./data/custom_generated_testset.csv") 