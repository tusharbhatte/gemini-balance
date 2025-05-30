"""
用户友好错误消息处理模块
提供将API错误转换为用户友好消息的功能
"""

import re
import json
import os
from typing import Dict, Any, Optional
from app.config.config import settings
from app.log.logger import get_gemini_logger

logger = get_gemini_logger()


class UserFriendlyErrorHandler:
    """用户友好错误处理器"""
    
    # 默认错误消息映射
    DEFAULT_ERROR_MESSAGES = {
        # HTTP状态码映射
        400: "请求参数错误，请检查您的输入",
        401: "API密钥无效或已过期，请联系管理员",
        403: "访问被拒绝，权限不足",
        404: "请求的资源不存在",
        408: "请求超时，请稍后重试",
        429: "请求过于频繁，请稍后重试", 
        500: "远程服务出现问题，请稍后重试",
        502: "网关错误，服务暂时不可用",
        503: "服务暂时不可用，请稍后重试",
        504: "网关超时，请稍后重试"
    }
    
    # 错误关键词映射
    ERROR_KEYWORD_MESSAGES = {
        "INTERNAL": "远程服务内部错误",
        "QUOTA_EXCEEDED": "API配额已超限",
        "PERMISSION_DENIED": "权限被拒绝",
        "INVALID_ARGUMENT": "请求参数无效",
        "DEADLINE_EXCEEDED": "请求超时",
        "RESOURCE_EXHAUSTED": "资源已耗尽",
        "UNAUTHENTICATED": "身份验证失败",
        "UNAVAILABLE": "服务暂时不可用",
        "NOT_FOUND": "请求的资源不存在",
        "ALREADY_EXISTS": "资源已存在",
        "CANCELLED": "请求被取消",
        "DATA_LOSS": "数据丢失",
        "UNKNOWN": "未知错误"
    }

    def __init__(self):
        self._custom_mappings = None
        self._load_custom_mappings()

    def _load_custom_mappings(self) -> None:
        """
        加载自定义错误消息映射
        支持从环境变量和配置文件中加载
        """
        self._custom_mappings = {}
        
        # 1. 从配置对象中加载
        if hasattr(settings, 'CUSTOM_ERROR_MAPPINGS') and settings.CUSTOM_ERROR_MAPPINGS:
            self._custom_mappings.update(settings.CUSTOM_ERROR_MAPPINGS)
            logger.info(f"Loaded {len(settings.CUSTOM_ERROR_MAPPINGS)} custom error mappings from config")
        
        # 2. 从环境变量中加载（格式：CUSTOM_ERROR_MAPPING_1=key1:value1）
        env_mappings = {}
        for env_key, env_value in os.environ.items():
            if env_key.startswith('CUSTOM_ERROR_MAPPING_'):
                try:
                    if ':' in env_value:
                        key, value = env_value.split(':', 1)  # 只分割第一个冒号
                        env_mappings[key.strip()] = value.strip()
                    else:
                        logger.warning(f"Invalid format for environment variable {env_key}: {env_value}")
                except Exception as e:
                    logger.error(f"Error parsing environment variable {env_key}: {e}")
        
        if env_mappings:
            self._custom_mappings.update(env_mappings)
            logger.info(f"Loaded {len(env_mappings)} custom error mappings from environment variables")
        
        # 3. 从环境变量JSON格式加载（格式：CUSTOM_ERROR_MAPPINGS_JSON={"key1":"value1","key2":"value2"})
        json_mappings_env = os.environ.get('CUSTOM_ERROR_MAPPINGS_JSON')
        if json_mappings_env:
            try:
                json_mappings = json.loads(json_mappings_env)
                if isinstance(json_mappings, dict):
                    self._custom_mappings.update(json_mappings)
                    logger.info(f"Loaded {len(json_mappings)} custom error mappings from JSON environment variable")
                else:
                    logger.warning("CUSTOM_ERROR_MAPPINGS_JSON is not a valid JSON object")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing CUSTOM_ERROR_MAPPINGS_JSON: {e}")
        
        logger.info(f"Total custom error mappings loaded: {len(self._custom_mappings)}")

    def _find_best_custom_match(self, message: str) -> Optional[str]:
        """
        在错误消息中查找最佳的自定义匹配
        
        匹配规则：
        1. 如果多个key同时匹配，以最长的key为准
        2. 如果key长度相等，按字典顺序取第一个匹配的key
        
        Args:
            message: 错误消息文本
            
        Returns:
            匹配的自定义消息，如果没有匹配则返回None
        """
        if not self._custom_mappings:
            return None
        
        matches = []
        message_lower = message.lower()
        
        # 查找所有匹配的键
        for key, value in self._custom_mappings.items():
            if key.lower() in message_lower:
                matches.append((key, value))
        
        if not matches:
            return None
        
        # 按匹配规则排序：先按长度降序，再按字典序升序
        matches.sort(key=lambda x: (-len(x[0]), x[0]))
        
        best_match = matches[0]
        logger.debug(f"Found custom error mapping match: '{best_match[0]}' -> '{best_match[1]}'")
        
        return best_match[1]

    @classmethod
    def extract_error_info(cls, error_content: str) -> Dict[str, Any]:
        """
        从错误内容中提取错误信息
        
        Args:
            error_content: 原始错误内容
            
        Returns:
            包含错误信息的字典
        """
        error_info = {
            "status_code": None,
            "original_message": "",
            "error_type": "UNKNOWN",
            "details": None
        }
        
        try:
            # 尝试解析JSON格式的错误
            if error_content.strip().startswith('{'):
                error_data = json.loads(error_content)
                
                # 处理Google API标准错误格式
                if "error" in error_data:
                    error_obj = error_data["error"]
                    error_info["status_code"] = error_obj.get("code")
                    error_info["original_message"] = error_obj.get("message", "")
                    error_info["error_type"] = error_obj.get("status", "UNKNOWN")
                    error_info["details"] = error_obj.get("details")
                
                # 处理OpenAI API错误格式
                elif "message" in error_data:
                    error_info["original_message"] = error_data.get("message", "")
                    error_info["error_type"] = error_data.get("type", "UNKNOWN")
                    
            else:
                # 处理非JSON格式的错误
                error_info["original_message"] = error_content
                
                # 尝试从错误消息中提取状态码
                status_match = re.search(r"status code (\d+)", error_content)
                if status_match:
                    error_info["status_code"] = int(status_match.group(1))
                
                # 尝试识别错误类型
                for keyword, error_type in cls.ERROR_KEYWORD_MESSAGES.items():
                    if keyword in error_content.upper():
                        error_info["error_type"] = keyword
                        break
                        
        except json.JSONDecodeError:
            # JSON解析失败，使用原始错误内容
            error_info["original_message"] = error_content
            logger.warning(f"Failed to parse error content as JSON: {error_content}")
        except Exception as e:
            logger.error(f"Error parsing error content: {e}")
            error_info["original_message"] = error_content
            
        return error_info

    def create_user_friendly_message(self, error_info: Dict[str, Any]) -> str:
        """
        创建用户友好的错误消息
        
        Args:
            error_info: 错误信息字典
            
        Returns:
            用户友好的错误消息
        """
        status_code = error_info.get("status_code")
        error_type = error_info.get("error_type", "UNKNOWN")
        original_message = error_info.get("original_message", "")
        
        # 1. 首先检查自定义错误映射
        custom_message = self._find_best_custom_match(original_message)
        if custom_message:
            return custom_message
        
        # 2. 然后尝试根据状态码获取消息
        if status_code and status_code in self.DEFAULT_ERROR_MESSAGES:
            base_message = self.DEFAULT_ERROR_MESSAGES[status_code]
        # 3. 最后尝试根据错误类型获取消息
        elif error_type in self.ERROR_KEYWORD_MESSAGES:
            base_message = self.ERROR_KEYWORD_MESSAGES[error_type]
        else:
            base_message = "调用远程服务出现问题"
        
        # 构建完整的用户友好消息
        if original_message:
            # 清理原始消息，移除技术性内容
            cleaned_message = self._clean_technical_message(original_message)
            if cleaned_message:
                user_message = f"{base_message}: {cleaned_message}"
            else:
                user_message = base_message
        else:
            user_message = base_message
            
        return user_message

    @classmethod
    def _clean_technical_message(cls, message: str) -> str:
        """
        清理技术性消息内容，保留用户可理解的部分
        
        Args:
            message: 原始技术消息
            
        Returns:
            清理后的消息
        """
        # 移除URL和技术链接
        message = re.sub(r'https?://[^\s]+', '', message)
        
        # 移除技术性的错误代码格式
        message = re.sub(r'\b[A-Z_]{2,}\b', '', message)
        
        # 移除括号中的技术内容
        message = re.sub(r'\([^)]*\)', '', message)
        
        # 清理多余的空格
        message = re.sub(r'\s+', ' ', message).strip()
        
        # 如果消息太短或为空，返回空字符串
        if len(message) < 10:
            return ""
            
        return message

    def handle_api_error(self, error_content: str, include_original: bool = False) -> Dict[str, Any]:
        """
        处理API错误，返回用户友好的错误响应
        
        Args:
            error_content: 原始错误内容
            include_original: 是否包含原始错误信息（用于调试）
            
        Returns:
            标准化的错误响应
        """
        error_info = self.extract_error_info(error_content)
        user_message = self.create_user_friendly_message(error_info)
        
        response = {
            "error": {
                "code": error_info.get("status_code", 500),
                "message": user_message,
                "status": "FAILED"
            }
        }
        
        # 如果启用调试模式或明确要求，包含原始错误信息
        if include_original or getattr(settings, 'DEBUG_MODE', False):
            response["error"]["original_error"] = {
                "message": error_info.get("original_message"),
                "type": error_info.get("error_type"),
                "details": error_info.get("details")
            }
            
        return response

    def reload_custom_mappings(self) -> None:
        """
        重新加载自定义错误映射配置
        可用于运行时更新配置
        """
        logger.info("Reloading custom error mappings...")
        self._load_custom_mappings()


# 全局实例
user_friendly_error_handler = UserFriendlyErrorHandler()