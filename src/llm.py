import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Callable, Tuple

from openai import OpenAI, AsyncOpenAI
from anthropic import AsyncAnthropic, Anthropic

from src.base_llm import APIGenerativeLLMModel
from src.llm_tools.response_parsing import parse_response_text

OPEN_AI_MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.150 / 1e6, "cached_input": 0.075 / 1e6, "output": 0.600 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "cached_input": 1.25 / 1e6, "output": 10.00 / 1e6},
    "gpt-4.5": {"input": 75.00 / 1e6, "cached_input": 37.50 / 1e6, "output": 150.00 / 1e6},
    "o1": {"input": 15.00 / 1e6, "cached_input": 7.50 / 1e6, "output": 60.00 / 1e6},
    "o1-mini": {"input": 1.10 / 1e6, "cached_input": 0.55 / 1e6, "output": 4.40 / 1e6},
    "o3-mini": {"input": 1.10 / 1e6, "cached_input": 0.55 / 1e6, "output": 4.40 / 1e6},
}

class OpenAIModel(APIGenerativeLLMModel):
    _api_key_name = "OPENAI_API_KEY"

    def __init__(self, model_name:str, api_key: Optional[str]=None):
        api_key = api_key or os.environ[self._api_key_name]
        super().__init__(model_name, api_key)

    def _get_response_text(self, model_response) -> str:
        return model_response.choices[0].message.content
    
    def _generate(self, client: Union[OpenAI, AsyncOpenAI], messages: List[Dict[str, str]], **kwargs)  -> Tuple[str, Dict[str, Any]]:
        
        response = client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            **kwargs
        )

        response_text = self._get_response_text(response)

        meta = {
            "cost": self.calculate_cost(response),
            "usage": response.usage,
            "finish_reason": response.choices[0].finish_reason,
        }

        try:
            parsed_response = parse_response_text(response_text)
        except:
            parsed_response = None

        
        return parsed_response or response_text, meta
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """

        kwargs examples
            reasoning_effort="medium",
            temperature=1.0
            
        """
        client = OpenAI(api_key=self.api_key)
        return self._generate(client, messages, **kwargs)

    async def a_generate(
        self,
        messages: List[Dict[str, str]],
        callback:Optional[Callable]=None,
        **kwargs
    ) -> str:
        client = AsyncOpenAI(api_key=self.api_key)
        
        response_text, meta = self._generate(client, messages, **kwargs)
    
        if callback is not None:
            callback()
            
        return response_text, meta

        
    def calculate_cost(self, model_response) -> float:
        usage = model_response.usage
        costs = OPEN_AI_MODEL_PRICING[self.model_name]

        if hasattr(usage, 'cache_read_input_tokens'):
            cached_tokens = usage.cache_read_input_tokens
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
        else:
            cached_tokens = usage.prompt_tokens_details.cached_tokens
            input_tokens = usage.prompt_tokens - cached_tokens
            output_tokens = usage.completion_tokens  # includes reasoning tokens

    
        return (
            costs['input'] * input_tokens + 
            costs['cached_input'] * cached_tokens + 
            costs['output'] * output_tokens
        )
    

class AnthropicModel(APIGenerativeLLMModel):
    _api_key_name = "ANTHROPIC_API_KEY"

    def __init__(self, model_name:str, api_key: Optional[str]=None):
        api_key = api_key or os.environ[self._api_key_name]
        super().__init__(model_name, api_key)

    def _get_response_text(self, response):
        return response.content[-1].text
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """

        kwargs examples
            thinking=True
            
        """
        client = Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model = self.model_name,
            messages = messages,
            max_tokens = 5_000,
            **kwargs
        )

        response_text = self._get_response_text(response)

        meta = {
            # "cost": self.calculate_cost(response),
            # "usage": response.usage,
            "finish_reason": response.stop_reason,
        }
        
        try:
            parsed_response = parse_response_text(response_text)
        except:
            parsed_response = None

        return parsed_response or response_text, meta

    async def a_generate(
        self,
        messages: List[Dict[str, str]],
        callback:Optional[Callable]=None,
        **kwargs
    ) -> str:
        # TODO
        pass

    def calculate_cost(self, model_response) -> float:
        # TODO
        pass


class LLMChat:
    def __init__(self, model, system_prompt):
        self.model = model
        self.chat_history = []

        self.system_prompt = system_prompt

        if isinstance(self.model, OpenAIModel):
            self.chat_history.append({
                "role":"system",
                "content": system_prompt
            })

    def query(self, user_message, **kwargs):
        self.chat_history.append({"role": "user", "content": user_message})
        # print(self.chat_history)
        if isinstance(self.model, OpenAIModel):
            response_text, meta = self.model.generate(self.chat_history, **kwargs)
        else:
            response_text, meta = self.model.generate(self.chat_history, system=self.system_prompt, **kwargs)

        self.chat_history.append({"role": "assistant", "content": response_text})
        
        return response_text