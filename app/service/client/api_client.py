# app/services/chat/api_client.py

from typing import Dict, Any, AsyncGenerator, Optional
import httpx
import json
import random
from abc import ABC, abstractmethod
from app.config.config import settings
from app.log.logger import get_api_client_logger
from app.handler.user_friendly_errors import user_friendly_error_handler

DEFAULT_TIMEOUT = 30
logger = get_api_client_logger()


class ApiErrorWithResponse(Exception):
    """包含友好错误响应的API异常"""
    
    def __init__(self, message: str, status_code: int, error_response: Dict[str, Any]):
        super().__init__(message)
        self.status_code = status_code
        self.error_response = error_response


class ApiClient(ABC):
    """API客户端基类"""

    @abstractmethod
    async def generate_content(self, payload: Dict[str, Any], model: str, api_key: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def stream_generate_content(self, payload: Dict[str, Any], model: str, api_key: str) -> AsyncGenerator[str, None]:
        pass

    def _handle_api_error(self, status_code: int, error_content: str) -> Dict[str, Any]:
        """
        统一处理API错误，生成友好错误响应并抛出异常
        
        Args:
            status_code: HTTP状态码
            error_content: 原始错误内容
            
        Returns:
            不会实际返回，总是抛出异常
            
        Raises:
            ApiErrorWithResponse: 包含友好错误响应的异常
        """
        # 生成友好的错误响应
        if settings.USER_FRIENDLY_ERRORS_ENABLED:
            # 使用用户友好错误处理器
            friendly_response = user_friendly_error_handler.handle_api_error(
                error_content, 
                include_original=settings.INCLUDE_TECHNICAL_DETAILS
            )
        else:
            # 返回原始错误信息的标准化格式
            friendly_response = {
                "error": {
                    "code": status_code,
                    "message": error_content,
                    "status": "FAILED"
                }
            }
        
        # 创建包含友好响应的异常
        error_exception = ApiErrorWithResponse(
            message=f"API call failed with status {status_code}: {error_content}",
            status_code=status_code,
            error_response=friendly_response
        )
        
        logger.error(f"API error occurred: {error_content}")
        raise error_exception


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
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/models?key={api_key}&pageSize=1000"
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
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")
            
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/models/{model}:generateContent?key={api_key}"
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                error_content = response.text
                return self._handle_api_error(response.status_code, error_content)
            return response.json()

    async def stream_generate_content(self, payload: Dict[str, Any], model: str, api_key: str) -> AsyncGenerator[str, None]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        model = self._get_real_model(model)
        
        proxy_to_use = None
        if settings.PROXIES:
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
            async with client.stream(method="POST", url=url, json=payload) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    error_msg = error_content.decode("utf-8")
                    # 对于流式响应，我们同时记录错误并返回友好错误响应
                    logger.error(f"API error occurred: {error_msg}")
                    
                    # 生成友好的错误响应
                    if settings.USER_FRIENDLY_ERRORS_ENABLED:
                        friendly_response = user_friendly_error_handler.handle_api_error(
                            error_msg, 
                            include_original=settings.INCLUDE_TECHNICAL_DETAILS
                        )
                    else:
                        friendly_response = {
                            "error": {
                                "code": response.status_code,
                                "message": error_msg,
                                "status": "FAILED"
                            }
                        }
                    
                    # 以SSE格式返回错误，但不抛出异常（因为这会中断生成器）
                    yield f"data: {json.dumps(friendly_response)}\n\n"
                    return
                async for line in response.aiter_lines():
                    yield line


class OpenaiApiClient(ApiClient):
    """OpenAI API客户端"""

    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        
    async def get_models(self, api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)

        proxy_to_use = None
        if settings.PROXIES:
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                error_content = response.text
                return self._handle_api_error(response.status_code, error_content)
            return response.json()

    async def generate_content(self, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        logger.info(f"settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY: {settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY}")
        proxy_to_use = None
        if settings.PROXIES:
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                error_content = response.text
                return self._handle_api_error(response.status_code, error_content)
            return response.json()

    async def stream_generate_content(self, payload: Dict[str, Any], api_key: str) -> AsyncGenerator[str, None]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        proxy_to_use = None
        if settings.PROXIES:
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}"}
            async with client.stream(method="POST", url=url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    error_msg = error_content.decode("utf-8")
                    error_response = self._handle_api_error(response.status_code, error_msg)
                    # 对于流式响应，我们需要以SSE格式返回错误
                    yield f"data: {json.dumps(error_response)}\n\n"
                    return
                async for line in response.aiter_lines():
                    yield line
    
    async def create_embeddings(self, input: str, model: str, api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)
        
        proxy_to_use = None
        if settings.PROXIES:
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

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
                return self._handle_api_error(response.status_code, error_content)
            return response.json()
                    
    async def generate_images(self, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        timeout = httpx.Timeout(self.timeout, read=self.timeout)

        proxy_to_use = None
        if settings.PROXIES:
            if settings.PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY:
                proxy_to_use = settings.PROXIES[hash(api_key) % len(settings.PROXIES)]
            else:
                proxy_to_use = random.choice(settings.PROXIES)
            logger.info(f"Using proxy for getting models: {proxy_to_use}")

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_to_use) as client:
            url = f"{self.base_url}/openai/images/generations"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                error_content = response.text
                return self._handle_api_error(response.status_code, error_content)
            return response.json()