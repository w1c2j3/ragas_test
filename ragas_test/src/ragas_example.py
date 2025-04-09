"""
RAG评估示例代码
使用ragas库进行RAG系统的评估
"""

import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_entities_recall,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset
import json
import numpy as np
from datetime import datetime
from custom_api_client import LangchainCustomLLMWrapper

# 加载环境变量
load_dotenv()

def simple_evaluation(dataset_path, output_dir="evaluation_results"):
    """
    执行简单的RAG评估
    
    Args:
        dataset_path (str): 数据集路径
        output_dir (str): 输出目录
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        print(f"加载数据集: {dataset_path}")
        dataset = Dataset.load_from_disk(dataset_path)
        
        # 初始化API客户端
        llm = LangchainCustomLLMWrapper()
        
        # 定义评估指标
        metrics = [
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
            context_entities_recall,
            answer_correctness,
            answer_similarity
        ]
        
        # 执行评估
        print("开始评估...")
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm
        )
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"rag_eval_{timestamp}.txt")
        
        # 保存评估结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RAG评估结果\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入每个样本的评估结果
            for i in range(len(dataset)):
                f.write(f"样本 {i+1}:\n")
                f.write(f"问题: {dataset[i]['question']}\n")
                f.write(f"上下文: {dataset[i]['contexts'][0]}\n")
                f.write(f"回答: {dataset[i]['answer']}\n")
                f.write(f"参考答案: {dataset[i]['ground_truths'][0]}\n")
                
                # 写入评估指标
                for metric in metrics:
                    metric_name = metric.__name__
                    if metric_name in result:
                        f.write(f"{metric_name}: {result[metric_name][i]}\n")
                
                f.write("\n" + "-" * 50 + "\n")
            
            # 写入总体统计信息
            f.write("\n总体统计:\n")
            for metric in metrics:
                metric_name = metric.__name__
                if metric_name in result:
                    scores = result[metric_name]
                    avg_score = np.mean(scores)
                    f.write(f"{metric_name} 平均值: {avg_score:.4f}\n")
        
        print(f"评估完成，结果已保存到: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        raise

def test_api_connection():
    """
    测试API连接是否正常
    """
    try:
        llm = LangchainCustomLLMWrapper()
        response = llm._call("测试API连接")
        print("API连接测试成功!")
        print(f"响应: {response}")
        return True
    except Exception as e:
        print(f"API连接测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 测试API连接
    if test_api_connection():
        # 执行评估
        dataset_path = "data/msmarco_for_ragas"
        simple_evaluation(dataset_path) 