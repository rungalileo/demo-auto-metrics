import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Callable, Tuple
import asyncio
from tqdm.auto import trange


class GenerativeModelBase(ABC):

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    async def a_generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(self, prompts: List[str], async_mode: bool = False, **kwargs) -> List[str]:
        pass


class APIGenerativeLLMModel(GenerativeModelBase):
    def __init__(self, model_name, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def calculate_cost(self, tokens_used: int) -> float:
        raise NotImplementedError

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        if not prompt and not messages:
            raise ValueError("Either `prompt` or `messages` must be set")

        if prompt:
            messages = [{
                "role": "user",
                "content": prompt,
            }]

        return self.generate(messages, **kwargs)

    async def batch_generate(
        self,
        prompts: Optional[List[str]]=None,
        messages: Optional[List[Dict[str, str]]]=None,
        **kwargs
    ) -> List[str]:
        
        if not prompts and not messages:
            raise ValueError("Either `prompts` or `messages` must be set")
            
        if prompts:
            messages = [
                [{"role": "user", "content": prompt}]
                for prompt in prompts
            ]

        pbar = trange(len(messages))
        
        coroutines = [self.a_generate(message, callback=pbar.update) for message in messages]
        result = await asyncio.gather(*coroutines)
        pbar.close()
        
        return result