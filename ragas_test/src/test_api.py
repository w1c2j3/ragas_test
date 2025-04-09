"""
简单的API测试脚本，验证自定义API连接是否正常
"""

import json
import time
import os
from dotenv import load_dotenv
from custom_api_client import CustomLLMClient

# 加载环境变量
load_dotenv()

class ApiTester:
    """API测试类，封装测试逻辑"""
    
    def __init__(self):
        """初始化测试客户端"""
        # 获取API密钥
        api_key = os.getenv("CUSTOM_API_KEY", "")
        self.api_url = "https://api.ppai.pro/v1/chat/completions"
        self.model_name = "gpt-4o-2024-11-20"
        
        # 创建统一的API客户端
        self.client = CustomLLMClient(
            api_url=self.api_url,
            api_key=api_key,
            model_name=self.model_name
        )
    
    def test_normal_call(self):
        """测试普通（非流式）API调用"""
        print("测试非流式API调用...")
        try:
            start_time = time.time()
            response = self.client._call("Say 'Hello, API test is working!' in Chinese")
            end_time = time.time()
            
            print(f"响应时间: {end_time - start_time:.2f} 秒")
            print(f"API响应内容: {response}")
            print("非流式API测试成功!")
            return True
        except Exception as e:
            print(f"非流式API测试失败: {e}")
            return False
    
    def test_stream_call(self):
        """测试流式API调用"""
        print("\n测试流式API调用...")
        try:
            start_time = time.time()
            full_text = ""
            
            print("流式响应开始接收:")
            print("-" * 40)
            
            # 使用generator获取流式响应
            for chunk in self.client.stream_chat("Count from 1 to 5 in Chinese, one number per line."):
                if chunk:
                    full_text += chunk
                    print(chunk, end="", flush=True)
            
            end_time = time.time()
            print("\n" + "-" * 40)
            print(f"流式响应总时间: {end_time - start_time:.2f} 秒")
            print(f"接收到的完整文本长度: {len(full_text)} 字符")
            print("流式API测试成功!")
            return True
        except Exception as e:
            print(f"流式API测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始API测试...\n")
        
        # 测试非流式API
        normal_test_success = self.test_normal_call()
        
        # 测试流式API
        stream_test_success = self.test_stream_call()
        
        # 总结测试结果
        print("\nAPI测试结果总结:")
        print(f"- 非流式API测试: {'成功' if normal_test_success else '失败'}")
        print(f"- 流式API测试: {'成功' if stream_test_success else '失败'}")
        
        if normal_test_success and stream_test_success:
            print("\n所有API测试通过，可以继续使用ragas进行评估!")
            return True
        else:
            print("\n一些API测试失败，请检查API配置和连接!")
            return False

if __name__ == "__main__":
    # 创建测试器并运行所有测试
    tester = ApiTester()
    tester.run_all_tests() 