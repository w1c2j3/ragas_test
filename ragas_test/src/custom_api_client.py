"""
定义一个自定义的LLM客户端，使用用户提供的API配置
"""

import json
import requests
import os
from typing import List, Dict, Any, Optional, Union, Generator
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class CustomLLMClient:
    """自定义LLM客户端，使用第三方API"""
    
    def __init__(self, api_url="https://api.ppai.pro/v1/chat/completions", api_key=None, model_name="gpt-4o-2024-11-20"):
        """初始化LLM客户端"""
        self.api_url = api_url
        # 使用环境变量中的API密钥（如果没有提供）
        self.api_key = api_key or os.getenv("CUSTOM_API_KEY", "")
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _prepare_request_data(self, prompt: str, stream: bool = False) -> Dict:
        """准备请求数据，避免代码重复"""
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
    
    def _call(self, prompt: str, **kwargs) -> str:
        """调用API获取非流式响应"""
        data = self._prepare_request_data(prompt, stream=False)
        
        print(f"发送请求到 {self.api_url}...")
        print(f"使用模型: {self.model_name}")
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            print("请求成功!")
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            return f"API调用失败: {str(e)}"
    
    def stream_chat(self, prompt: str, **kwargs) -> Generator[str, None, str]:
        """调用API获取流式响应"""
        data = self._prepare_request_data(prompt, stream=True)
        
        print(f"发送流式请求到 {self.api_url}...")
        print(f"使用模型: {self.model_name}")
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=data, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data_str = line[5:].strip()
                        if data_str != "[DONE]":
                            try:
                                data_json = json.loads(data_str)
                                if "choices" in data_json and len(data_json["choices"]) > 0:
                                    delta = data_json["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        full_response += content
                                        yield content
                            except json.JSONDecodeError:
                                print(f"无法解析响应: {data_str}")
            
            return full_response
        except requests.exceptions.RequestException as e:
            print(f"API流式请求失败: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            yield f"API流式调用失败: {str(e)}"


class LangchainCustomLLMWrapper(LLM):
    """为Ragas评估提供的LangChain兼容LLM包装器"""
    
    api_url: str = "https://api.ppai.pro/v1/chat/completions"
    api_key: str = os.getenv("CUSTOM_API_KEY", "")
    model_name: str = "gpt-4o-2024-11-20"
    
    client: Optional[CustomLLMClient] = None
    
    def __init__(self):
        """初始化LLM客户端"""
        super().__init__()
        self.client = CustomLLMClient(self.api_url, self.api_key, self.model_name)
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "custom_llm"
    
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        """LangChain LLM _call方法实现"""
        if not self.client:
            self.client = CustomLLMClient(self.api_url, self.api_key, self.model_name)
        
        try:
            return self.client._call(prompt)
        except Exception as e:
            print(f"API调用失败: {e}")
            raise e
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数"""
        return {"model_name": self.model_name}


# 使用示例
if __name__ == "__main__":
    client = CustomLLMClient()
    response = client._call("Hello, how are you today?")
    print(f"普通响应: {response}")
    
    print("\n流式响应测试:")
    for chunk in client.stream_chat("Tell me a short story about a cat."):
        print(chunk, end="", flush=True) 