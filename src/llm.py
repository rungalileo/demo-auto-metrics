import os
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from pydantic import BaseModel

from openai import OpenAI
from anthropic import AsyncAnthropic, Anthropic

from src.prompt_templates import LLMOutputSchema, AutoMetricLLMOutput
from src.llm_tools.api_call import call_claude_3_7_sonnet_and_parse_with_retries
from src.llm_tools.response_parsing import parse_response_text

class OpenaiModel:
    def __init__(self, model_name: str="gpt-4o-mini", api_key:Optional[str]=None):
        self.model_name = model_name

        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif "OPENAI_API_KEY" in os.environ:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise KeyError("OPENAI_API_KEY is not set. Either set the env variable, or pass it in as an argument.")
    
    def generate_response(self, user_prompt:str, output_schema=Optional[LLMOutputSchema], temperature=1.0):
        """
        Args:
            user_prompt (str): test to query the LLM with
            output_schema (LLMOutputSchema): when provided will return parsed response
        
        Returns
            response
            cost
        """
        messages=[{
            "role": "user",
            "content": user_prompt,
        }]

        if output_schema:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format=output_schema,
            )
            return completion.choices[0].message.parsed
        else:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                messages=messages,
            )
            parsed = parse_response_text(completion.choices[0].message.content)
            return AutoMetricLLMOutput(**parsed)

class AnthropicModel:
    def __init__(self, api_key:Optional[str]=None):

        if api_key:
            self.client = Anthropic(api_key=api_key)
        elif "ANTHROPIC_API_KEY" in os.environ:
            self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            raise KeyError("ANTHROPIC_API_KEY is not set. Either set the env variable, or pass it in as an argument.")


    def generate_response(self, user_prompt:str, **kwargs):
        """
        Args:
            user_prompt (str): test to query the LLM with
            output_schema (LLMOutputSchema): when provided will return parsed response
        
        Returns
            response
            cost
        """
        messages=[{
            "role": "user",
            "content": user_prompt,
        }]
        
        parsed_response = call_claude_3_7_sonnet_and_parse_with_retries(messages, self.client, max_tries=1)
        if isinstance(parsed_response[0], str):
            parsed_response = parse_response_text(parsed_response[0])
        else:
            parsed_response = parsed_response[0]
        return AutoMetricLLMOutput(**parsed_response)