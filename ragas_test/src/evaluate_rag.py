"""
使用Ragas和MS MARCO数据集评估RAG系统的综合脚本
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AspectCritic
from src.custom_api_client import LangchainCustomLLMWrapper
import json

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAG系统评估工具")
    parser.add_argument("--samples", type=int, default=0, help="要评估的样本数量，0表示评估全部")
    parser.add_argument("--metrics", type=str, default="accuracy,completeness", help="要使用的评估指标，用逗号分隔")
    parser.add_argument("--dataset", type=str, default="data/msmarco_for_ragas", help="评估数据集的路径")
    parser.add_argument("--output", type=str, default="", help="评估结果输出文件路径，不指定则只输出到控制台")
    parser.add_argument("--json-output", type=str, default="", help="评估结果JSON格式输出文件路径，便于可视化")
    parser.add_argument("--batch-size", type=int, default=50, help="批处理大小，避免一次性评估太多样本")
    parser.add_argument("--visualize", action="store_true", help="评估完成后直接生成可视化结果")
    return parser.parse_args()

def create_metrics(metric_names):
    """创建评估指标"""
    metrics = []
    
    metrics_definitions = {
        "accuracy": "评估回答的准确性，验证回答是否与上下文内容一致，且不包含虚假信息。",
        "completeness": "评估回答的完整性，验证回答是否涵盖了上下文中的所有关键信息。",
        "relevance": "评估回答与问题的相关性，验证回答是否直接解答了用户的问题。",
        "coherence": "评估回答的连贯性，验证回答是否逻辑清晰且结构良好。",
        "conciseness": "评估回答的简洁性，验证回答是否简明扼要，不包含冗余信息。"
    }
    
    for name in metric_names:
        if name in metrics_definitions:
            metrics.append(AspectCritic(
                name=name,
                definition=metrics_definitions[name]
            ))
        else:
            print(f"警告: 未知的评估指标 '{name}'，已忽略")
    
    return metrics

def evaluate_dataset(dataset_path, sample_count, metrics, output_file=None, json_output=None, batch_size=50, visualize=False):
    """评估数据集"""
    # 设置输出目标
    original_stdout = sys.stdout
    file_out = None
    
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        file_out = open(output_file, 'w', encoding='utf-8')
    
    def log(message, to_console=True):
        """同时输出到文件和控制台"""
        if file_out:
            print(message, file=file_out, flush=True)
        if to_console:
            print(message, flush=True)
    
    try:
        # 输出评估信息头部
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"=========== RAG系统评估报告 ===========")
        log(f"评估时间: {current_time}")
        log(f"数据集路径: {dataset_path}")
        log(f"评估指标: {', '.join([m.name for m in metrics])}")
        log("=======================================\n")
        
        # 加载数据集
        if os.path.exists(dataset_path):
            log(f"加载数据集: {dataset_path}")
            dataset = Dataset.load_from_disk(dataset_path)
        else:
            log(f"错误: 数据集路径不存在: {dataset_path}")
            return
        
        # 检查和处理样本数量
        total_samples = len(dataset)
        if sample_count <= 0 or sample_count > total_samples:
            sample_count = total_samples
            log(f"将评估全部 {total_samples} 个样本")
        else:
            log(f"数据集总样本数: {total_samples}")
            log(f"将评估 {sample_count} 个样本")
        
        # 创建LLM客户端
        llm_wrapper = LangchainCustomLLMWrapper()
        log(f"使用API: {llm_wrapper.client.api_url}")
        log(f"使用模型: {llm_wrapper.client.model_name}")
        
        # 批量评估，避免一次性处理太多样本
        all_results = {}
        
        # 计时
        start_time = time.time()
        
        # 分批处理
        for i in range(0, sample_count, batch_size):
            batch_end = min(i + batch_size, sample_count)
            progress_msg = f"\n正在评估批次 {i//batch_size + 1}/{(sample_count+batch_size-1)//batch_size}: 样本 {i+1}-{batch_end}/{sample_count}"
            log(progress_msg)
            
            # 准备评估数据集
            test_dataset = dataset.select(range(i, batch_end))
            
            # 运行评估
            batch_start_time = time.time()
            try:
                # 评估前输出信息
                log(f"开始评估批次 {i//batch_size + 1}...", to_console=True)
                
                result = evaluate(
                    dataset=test_dataset,
                    metrics=metrics,
                    llm=llm_wrapper
                )
                
                # 提取结果
                if hasattr(result, '_scores_dict'):
                    scores_dict = result._scores_dict
                    for metric_name, scores in scores_dict.items():
                        if metric_name not in all_results:
                            all_results[metric_name] = []
                        all_results[metric_name].extend(scores)
                    
                    # 输出当前批次结果
                    log(f"\n批次 {i//batch_size + 1} 评估结果:", to_console=False)
                    for metric_name, scores in scores_dict.items():
                        log(f"{metric_name}: {scores}", to_console=False)
                    
                    # 在控制台输出简要结果
                    for metric_name, scores in scores_dict.items():
                        if scores:
                            valid_scores = [x for x in scores if x is not None and not np.isnan(x)]
                            if valid_scores:
                                avg = np.mean(valid_scores)
                                log(f"批次 {i//batch_size + 1} {metric_name} 平均分: {avg:.4f}")
                else:
                    log("无法获取结果分数字典")
                    
            except Exception as e:
                log(f"批次 {i//batch_size + 1} 评估过程中出现错误: {e}")
                if file_out:
                    import traceback
                    traceback.print_exc(file=file_out)
                log("继续评估下一批次...")
                continue
            
            batch_time = time.time() - batch_start_time
            log(f"批次 {i//batch_size + 1} 耗时: {batch_time:.2f} 秒")
            
            # 输出总进度
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + batch_size)) * (sample_count - (i + batch_size)) if i + batch_size < sample_count else 0
            log(f"总进度: {min((i + batch_size), sample_count)}/{sample_count} 样本, 已用时: {elapsed:.2f}秒, 预计剩余: {remaining:.2f}秒")
        
        # 评估完成，输出统计结果
        total_time = time.time() - start_time
        log("\n=========== 评估完成 ===========")
        log(f"总耗时: {total_time:.2f} 秒")
        log(f"评估样本数: {sample_count}")
        log(f"每样本平均耗时: {total_time/sample_count:.2f} 秒")
        log("=================================\n")
        
        # 处理汇总结果
        if all_results:
            log("\n========== 汇总结果 ==========")
            log("\n原始分数:", to_console=False)
            for metric_name, scores in all_results.items():
                log(f"{metric_name}: {scores}", to_console=False)
            
            log("\n统计摘要:")
            for metric_name, scores in all_results.items():
                if isinstance(scores, list) and len(scores) > 0:
                    valid_scores = [x for x in scores if x is not None and not np.isnan(x)]
                    if valid_scores:
                        avg = np.mean(valid_scores)
                        median = np.median(valid_scores)
                        min_val = np.min(valid_scores)
                        max_val = np.max(valid_scores)
                        log(f"{metric_name}:")
                        log(f"  平均值: {avg:.4f}")
                        log(f"  中位数: {median:.4f}")
                        log(f"  最小值: {min_val:.4f}")
                        log(f"  最大值: {max_val:.4f}")
                        log(f"  有效样本数: {len(valid_scores)}/{len(scores)}")
                    else:
                        log(f"{metric_name}: 无有效分数")
                else:
                    log(f"{metric_name}: {scores}")
            log("===============================")
            
            # 保存JSON格式结果
            if json_output:
                json_dir = os.path.dirname(os.path.abspath(json_output))
                os.makedirs(json_dir, exist_ok=True)
                
                # 将浮点数和NaN值转换为可JSON序列化的格式
                json_results = {}
                for metric, values in all_results.items():
                    json_results[metric] = [
                        float(v) if not np.isnan(v) and v is not None else None 
                        for v in values
                    ]
                
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, ensure_ascii=False, indent=2)
                
                log(f"JSON格式评估结果已保存到: {json_output}")
                
            # 如果需要，调用可视化脚本
            if visualize and (json_output or output_file):
                try:
                    log("生成可视化结果...")
                    
                    # 构建命令行参数
                    result_file = json_output if json_output else output_file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    viz_output = f"visualization_results/rag_viz_{timestamp}.html"
                    
                    # 导入可视化模块并调用
                    from src.visualize_results import create_visualization
                    viz_path = create_visualization(dataset_path, result_file, viz_output)
                    log(f"可视化结果已生成: {viz_path}")
                except Exception as e:
                    log(f"生成可视化时出错: {e}")
                
    except Exception as e:
        log(f"评估过程中出现错误: {e}")
        if file_out:
            import traceback
            traceback.print_exc(file=file_out)
        log("注意: 请确保API连接正常，并且API服务支持ragas所需的功能")
    
    finally:
        # 关闭文件
        if file_out:
            file_out.close()
            print(f"评估结果已保存到: {output_file}")

if __name__ == "__main__":
    args = parse_args()
    
    # 处理指标列表
    metric_names = [name.strip() for name in args.metrics.split(",")]
    metrics = create_metrics(metric_names)
    
    if not metrics:
        print("错误: 没有有效的评估指标")
        exit(1)
    
    # 如果没有指定输出文件，自动生成一个带时间戳的文件名
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        metrics_str = "_".join(metric_names)
        sample_str = "all" if args.samples <= 0 else str(args.samples)
        output_file = f"{output_dir}/rag_eval_{metrics_str}_{sample_str}_{timestamp}.txt"
    
    # 如果没有指定JSON输出文件，但需要可视化，则自动生成一个JSON输出文件
    json_output = args.json_output
    if not json_output and args.visualize:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_dir = "evaluation_results/json"
        os.makedirs(json_dir, exist_ok=True)
        metrics_str = "_".join(metric_names)
        sample_str = "all" if args.samples <= 0 else str(args.samples)
        json_output = f"{json_dir}/rag_eval_{metrics_str}_{sample_str}_{timestamp}.json"
    
    # 运行评估
    evaluate_dataset(args.dataset, args.samples, metrics, output_file, json_output, args.batch_size, args.visualize) 