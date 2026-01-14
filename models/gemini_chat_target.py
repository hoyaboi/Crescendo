from typing import List, Optional, Dict, Any
import asyncio
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")

from pyrit.models import ChatMessage, Message, MessagePiece, ChatMessageRole, PromptDataType, construct_response_from_request
from pyrit.prompt_target import PromptChatTarget


class GeminiChatTarget(PromptChatTarget):
    def __init__(
        self,
        model_name: str = "gemini-pro",
        api_key: Optional[str] = None,
        max_output_tokens: int = 2048,
    ):
        super().__init__()
        
        if api_key is None:
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self._model_name = model_name
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def _validate_request(self, *, message: Message) -> None:
        if not message or not message.message_pieces:
            raise ValueError("Message must have at least one message piece")
    
    def is_json_response_supported(self) -> bool:
        return False
    
    def _convert_chat_messages_to_gemini_format(self, messages: List[ChatMessage]) -> str:
        """
        Convert PyRIT ChatMessage list to Gemini prompt format.
        Gemini uses a simple text prompt format, so we concatenate messages.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.role
            content = msg.content
            
            if role == "system" or (isinstance(role, str) and role.lower() == "system"):
                # System messages are typically prepended
                prompt_parts.insert(0, f"System: {content}\n")
            elif role == "user" or (isinstance(role, str) and role.lower() == "user"):
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant" or (isinstance(role, str) and role.lower() == "assistant"):
                prompt_parts.append(f"Assistant: {content}\n")
        
        return "".join(prompt_parts)
    
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
        
        # Convert to Gemini format
        prompt = self._convert_chat_messages_to_gemini_format(chat_messages)
        
        try:
            # Run in executor to make synchronous call async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self._max_output_tokens,
                    )
                )
            )
            
            # Extract response text
            response_text = ""
            if response.text:
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Fallback: try to get text from candidates
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                response_text += part.text
            
            response_message = construct_response_from_request(
                request=request_piece,
                response_text_pieces=[response_text],
                response_type="text",
            )
            
            return [response_message]
            
        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {str(e)}") from e
