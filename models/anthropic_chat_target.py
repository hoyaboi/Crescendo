from typing import List, Optional, Dict, Any
import asyncio
try:
    from anthropic import Anthropic, AsyncAnthropic
except ImportError:
    raise ImportError("anthropic package is required. Install it with: pip install anthropic")

from pyrit.models import ChatMessage, Message, MessagePiece, ChatMessageRole, PromptDataType, construct_response_from_request
from pyrit.prompt_target import PromptChatTarget


class AnthropicChatTarget(PromptChatTarget):
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        super().__init__()
        
        if api_key is None:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self._model_name = model_name
        self._api_key = api_key
        self._max_tokens = max_tokens
        
        self._client = Anthropic(api_key=api_key)
        self._async_client = AsyncAnthropic(api_key=api_key)
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def _validate_request(self, *, message: Message) -> None:
        if not message or not message.message_pieces:
            raise ValueError("Message must have at least one message piece")
    
    def is_json_response_supported(self) -> bool:
        return False
    
    def _convert_chat_messages_to_anthropic_format(self, messages: List[ChatMessage]) -> tuple[List[Dict[str, Any]], Optional[str]]:
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            role = msg.role
            if role == "system" or (isinstance(role, str) and role.lower() == "system"):
                system_prompt = msg.content
            elif role == "assistant" or (isinstance(role, str) and role.lower() == "assistant"):
                anthropic_messages.append({"role": "assistant", "content": msg.content})
            elif role == "user" or (isinstance(role, str) and role.lower() == "user"):
                anthropic_messages.append({"role": "user", "content": msg.content})
        
        return anthropic_messages, system_prompt
    
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        self._validate_request(message=message)
        
        request_piece: MessagePiece = message.message_pieces[0]
        conversation_id = request_piece.conversation_id
        
        chat_messages = []
        if conversation_id:
            try:
                chat_messages = list(self._memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id))
            except Exception:
                chat_messages = []
        
        request_chat_message = request_piece.to_chat_message()
        chat_messages.append(request_chat_message)
        
        anthropic_messages, system_prompt = self._convert_chat_messages_to_anthropic_format(chat_messages)
        
        try:
            kwargs = {
                "model": self._model_name,
                "max_tokens": self._max_tokens,
                "messages": anthropic_messages,
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = await self._async_client.messages.create(**kwargs)
            
            # 응답 추출
            response_text = ""
            if response.content:
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        response_text += content_block.text
                    elif isinstance(content_block, dict) and 'text' in content_block:
                        response_text += content_block['text']
            
            response_message = construct_response_from_request(
                request=request_piece,
                response_text_pieces=[response_text],
                response_type="text",
            )
            
            return [response_message]
            
        except Exception as e:
            raise RuntimeError(f"Error calling Anthropic API: {str(e)}") from e
