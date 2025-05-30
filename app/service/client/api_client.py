# app/services/chat/api_client.py

from typing import Dict, Any, AsyncGenerator, Optional
import httpx
import random
from abc import ABC, abstractmethod
from app.config.config import settings
from app.log.logger import get_api_client_logger
from app.handler.user_friendly_errors import user_friendly_error_handler

DEFAULT_TIMEOUT = 30
logger = get_api_client_logger()

class ApiClient(ABC):
    """API客户端基类"""

    @abstractmethod
    async def generate_content(self, payload: Dict[str, Any], model: str, api_key: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def stream_generate_content(self, payload: Dict[str, Any], model: str, api_key: str) -> AsyncGenerator[str, None]:
        pass

    def _handle_api_error(self, status_code: int, error_content: str) -> Exception:
        """
        统一处理API错误，根据配置返回用户友好或技术性错误信息
        
        Args:
            status_code: HTTP状态码
            error_content: 原始错误内容
            
        Returns:
            处理后的异常对象
        """
        if settings.USER_FRIENDLY_ERRORS_ENABLED:
            # 使用用户友好错误处理器
            friendly_response = user_friendly_error_handler.handle_api_error(
                error_content, 
                include_original=settings.INCLUDE_TECHNICAL_DETAILS
            )
            
            # 从友好响应中提取消息
            friendly_message = friendly_response.get("error", {}).get("message", "调用远程服务出现问题")
            
            # 如果包含技术细节，添加到错误消息中
            if settings.INCLUDE_TECHNICAL_DETAILS and "original_error" in friendly_response.get("error", {}):
                original_msg = friendly_response["error"]["original_error"].get("message", "")
                if original_msg:
                    friendly_message = f"{friendly_message} (技术详情: {original_msg})"
            
            return Exception(f"API call failed with status code {status_code}, {friendly_message}")
        else:
            # 使用原始错误信息
            return Exception(f"API call failed with status code {status_code}, {error_content}")


class GeminiApiClient(ApiClient):
    """Gemini API客户端"""

    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    def _get_real_model(self, model: str) -> str:
        if model.endswith("-search"):
            model = model[:-7]
        if model.endswith("-image"):
            model = model[:-6]
        if model.endswith("-non-thinking"):
            model = model[:-13]
        if "-search" in model and "-non-thinking" in model:
            model = model[:-20]
        return model

    async def get_models(self, api_key: str) -> Optional[Dict[str, Any]]:
        """获取可用的 Gemini 模型列表"""
        timeout = httpx.Timeout(timeout=5)
        
        proxy_to_use = None
        if settings.PROXIES:
            proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/models?key={api_key}"
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"获取模型列表失败: {e.response.status_code}")
                logger.error(e.response.text)
                return None
            except httpx.RequestError as e:
                logger.error(f"请求模型列表失败: {e}")
                return None
            
    async def generate_content(self, payload: Dict[str, Any], model: str, api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        model = self._get_real_model(model)

        proxy_to_use = None
        if settings.PROXIES:
            proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy: {proxy_to_use}")
            
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/models/{model}:generateContent?key={api_key}"
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                error_content = response.text
                raise self._handle_api_error(response.status_code, error_content)
            return response.json()

    async def stream_generate_content(self, payload: Dict[str, Any], model: str, api_key: str) -> AsyncGenerator[str, None]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        model = self._get_real_model(model)
        
        proxy_to_use = None
        if settings.PROXIES:
            proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
            async with client.stream(method="POST", url=url, json=payload) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    error_msg = error_content.decode("utf-8")
                    raise self._handle_api_error(response.status_code, error_msg)
                async for line in response.aiter_lines():
                    yield line


class OpenaiApiClient(ApiClient):
    """OpenAI API客户端"""

    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        
    async def get_models(self, api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{self.base_url}/openai/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                error_content = response.text
                raise self._handle_api_error(response.status_code, error_content)
            return response.json()

    async def generate_content(self, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        
        proxy_to_use = None
        if settings.PROXIES:
            proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                error_content = response.text
                raise self._handle_api_error(response.status_code, error_content)
            return response.json()

    async def stream_generate_content(self, payload: Dict[str, Any], api_key: str) -> AsyncGenerator[str, None]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        
        proxy_to_use = None
        if settings.PROXIES:
            proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}"}
            async with client.stream(method="POST", url=url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    error_msg = error_content.decode("utf-8")
                    raise self._handle_api_error(response.status_code, error_msg)
                async for line in response.aiter_lines():
                    yield line
    
    async def create_embeddings(self, input: str, model: str, api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        
        proxy_to_use = None
        if settings.PROXIES:
            proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/embeddings"
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "input": input,
                "model": model,
            }
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                error_content = response.text
                raise self._handle_api_error(response.status_code, error_content)
            return response.json()
                    
    async def generate_images(self, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        
        proxy_to_use = None
        if settings.PROXIES:
            proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/images/generations"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                error_content = response.text
                raise self._handle_api_error(response.status_code, error_content)
            return response.json()