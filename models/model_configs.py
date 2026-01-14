from dataclasses import dataclass
from typing import Optional, Any
import os
import torch

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.prompt_target import HuggingFaceChatTarget


@dataclass
class ModelConfig:
    name: str
    model_type: str
    deployment_name: str
    api_key_env: str
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    
    def to_target(self):
        api_key = os.getenv(self.api_key_env)
        
        if not api_key and self.model_type != "huggingface":
            raise ValueError(f"API key not found: {self.api_key_env}")
        
        if self.model_type == "openai":
            endpoint = self.endpoint if self.endpoint else "https://api.openai.com/v1"
            return OpenAIChatTarget(
                model_name=self.deployment_name,
                api_key=api_key,
                endpoint=endpoint,
            )
        
        elif self.model_type == "huggingface":
            return HuggingFaceChatTarget(
                model_id=self.deployment_name,
                hf_access_token=api_key,
                use_cuda=True,
                trust_remote_code=True,
                max_new_tokens=256,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else None
            )
        
        # 사용자 정의 모델 타입은 여기에 추가
        # 예: remote-openai, anthropic 등
        
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Please implement custom model support in ModelConfig.to_target() method."
            )


# 사용 가능한 모델 프리셋
AVAILABLE_MODELS = {
    # OpenAI Models
    "gpt-4": ModelConfig(
        name="GPT-4",
        model_type="openai",
        deployment_name="gpt-4",
        api_key_env="OPENAI_API_KEY"
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        model_type="openai",
        deployment_name="gpt-4o",
        api_key_env="OPENAI_API_KEY"
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o-mini",
        model_type="openai",
        deployment_name="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY"
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5-Turbo",
        model_type="openai",
        deployment_name="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY"
    ),
    
    # Meta Models
    "llama-2-70b": ModelConfig(
        name="LLaMA-2-70b",
        model_type="huggingface",
        deployment_name="meta-llama/Llama-2-70b-chat-hf",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "llama-3-70b": ModelConfig(
        name="LLaMA-3-70b",
        model_type="huggingface",
        deployment_name="meta-llama/Meta-Llama-3-70B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "llama-2-7b": ModelConfig(
        name="LLaMA-2-7b",
        model_type="huggingface",
        deployment_name="meta-llama/Llama-2-7b-chat-hf",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "llama-3-8b": ModelConfig(
        name="LLaMA-3-8B",
        model_type="huggingface",
        deployment_name="meta-llama/Meta-Llama-3-8B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    
    # ============================================================================
    # 사용자 정의 모델 추가 예시
    # ============================================================================
    # 
    # Remote Server Models (vLLM 등 OpenAI 호환 API)
    # 아래 예시를 참고하여 직접 추가하세요:
    #
    # "llama-3-8b-remote": ModelConfig(
    #     name="LLaMA-3-8B-Remote",
    #     model_type="remote-openai",
    #     deployment_name="meta-llama/Meta-Llama-3-8B-Instruct",
    #     api_key_env="REMOTE_API_KEY",
    #     endpoint="REMOTE_SERVER_ENDPOINT"  # 또는 직접 URL: "http://your-server:port/v1"
    # ),
    #
    # Anthropic Claude Models
    # 아래 예시를 참고하여 직접 추가하세요:
    #
    # "claude-3-5-sonnet": ModelConfig(
    #     name="Claude-3.5-Sonnet",
    #     model_type="anthropic",
    #     deployment_name="claude-3-5-sonnet-20241022",
    #     api_key_env="ANTHROPIC_API_KEY"
    # ),
    #
    # 참고: custom model type을 사용하려면:
    # 1. ModelConfig.to_target() 메서드에 해당 타입 처리 로직 추가
    # 2. 필요한 경우 custom chat target 클래스 구현 (예: AnthropicChatTarget)
    # 3. AVAILABLE_MODELS에 모델 설정 추가
    #
    # ============================================================================
}
