"""
检查MS MARCO数据集的结构
"""

from datasets import load_dataset

def check_dataset_structure():
    """
    加载并输出MS MARCO数据集的结构
    """
    print("加载MS MARCO数据集...")
    dataset = load_dataset("ms_marco", "v2.1", split="train[:5]")
    
    print(f"\n数据集大小: {len(dataset)} 个样本")
    print(f"数据集列名: {dataset.column_names}")
    
    # 显示一个样本的内容
    sample = dataset[0]
    print("\n第一个样本结构:")
    for key in sample:
        if isinstance(sample[key], dict):
            print(f"{key}: (dict)")
            for sub_key in sample[key]:
                print(f"  {sub_key}: {type(sample[key][sub_key])}")
                # 如果是列表，显示第一个元素(如果有)
                if isinstance(sample[key][sub_key], list) and len(sample[key][sub_key]) > 0:
                    print(f"    第一个元素: {sample[key][sub_key][0]}")
        elif isinstance(sample[key], list):
            print(f"{key}: (list) 长度={len(sample[key])}")
            # 显示第一个元素(如果有)
            if len(sample[key]) > 0:
                print(f"  第一个元素: {sample[key][0]}")
        else:
            print(f"{key}: {sample[key]}")

if __name__ == "__main__":
    check_dataset_structure() 