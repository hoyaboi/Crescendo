from dataclasses import dataclass
from typing import Optional, Any
import os
import torch

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.prompt_target import HuggingFaceChatTarget
from .anthropic_chat_target import AnthropicChatTarget


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
        
        if not api_key and self.model_type not in ["remote-openai", "anthropic", "huggingface"]:
            raise ValueError(f"API key not found: {self.api_key_env}")
        
        if self.model_type == "openai":
            endpoint = self.endpoint if self.endpoint else "https://api.openai.com/v1"
            return OpenAIChatTarget(
                model_name=self.deployment_name,
                api_key=api_key,
                endpoint=endpoint,
            )
        
        elif self.model_type == "remote-openai":
            if self.endpoint and not self.endpoint.startswith("http"):
                endpoint = os.getenv(self.endpoint)
                if not endpoint:
                    raise ValueError(
                        f"endpoint is required for remote-openai model type. "
                        f"Set {self.endpoint} environment variable or configure endpoint in model config."
                    )
            else:
                endpoint = self.endpoint
                if not endpoint:
                    raise ValueError(f"endpoint is required for remote-openai model type")
            api_key_value = api_key if api_key else "dummy-key"
            return OpenAIChatTarget(
                model_name=self.deployment_name,
                api_key=api_key_value,
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
        
        elif self.model_type == "anthropic":
            return AnthropicChatTarget(
                model_name=self.deployment_name,
                api_key=api_key,
                max_tokens=4096
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


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
    
    # Remote Server Models
    "llama-3-8b-remote": ModelConfig(
        name="LLaMA-3-8B-Remote",
        model_type="remote-openai",
        deployment_name="meta-llama/Meta-Llama-3-8B-Instruct",
        api_key_env="REMOTE_API_KEY",
        endpoint="REMOTE_SERVER_ENDPOINT"
    ),
    "llama-2-7b-remote": ModelConfig(
        name="LLaMA-2-7B-Remote",
        model_type="remote-openai",
        deployment_name="meta-llama/Llama-2-7b-chat-hf",
        api_key_env="REMOTE_API_KEY",
        endpoint="REMOTE_SERVER_ENDPOINT"
    ),
    
    # Anthropic Claude Models
    "claude-3-5-sonnet": ModelConfig(
        name="Claude-3.5-Sonnet",
        model_type="anthropic",
        deployment_name="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY"
    ),
    "claude-3-opus": ModelConfig(
        name="Claude-3-Opus",
        model_type="anthropic",
        deployment_name="claude-3-opus-20240229",
        api_key_env="ANTHROPIC_API_KEY"
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude-3-Haiku",
        model_type="anthropic",
        deployment_name="claude-3-haiku-20240307",
        api_key_env="ANTHROPIC_API_KEY"
    ),
    "claude-sonnet-4": ModelConfig(
        name="Claude-Sonnet-4",
        model_type="anthropic",
        deployment_name="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY"
    ),
}
