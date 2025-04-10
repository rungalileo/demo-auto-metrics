from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum

from src.llm import OpenAIModel, AnthropicModel
from src.prompt_templates import AutoMetricLLMOutput, get_filled_prompt


class AppLabel(str, Enum):
    RAG = "Retriever Augmented Generation (RAG)"
    AGENT = "Agent"
    CODE_GEN = "Code Generation"
    MULTI_TURN = "Multi-turn dialogue system"

    def __str__(self) -> str:
        return self.value
    

@dataclass(frozen=True)
class MetricsGenerationConfig:
    model_name: str = "gpt-4o"
    app_description: Optional[str] = None
    app_labels: Optional[List[Union[str, AppLabel]]] = None
    api_key: Optional[str] = None

class MetricsGenerator:
    def __init__(self, config: MetricsGenerationConfig):
        """
        Args:
            model (str): model id to use for generation (e.g. "gpt-4o")
            config (DataGenerationConfig): generation config
        """
        self.config = config
        self.is_async = False
        if config.model_name == "gpt-4o":
            self.model = OpenAIModel(config.model_name, api_key=config.api_key) # assume its an openai model for now
        elif 'claude' in config.model_name:
            self.model = AnthropicModel('claude-3-7-sonnet-20250219', api_key=config.api_key)
        
    
    def generate(self):
        
        filled_prompt = get_filled_prompt(
            user_app_description = self.config.app_description,
            app_labels = ",".join(self.config.app_labels),
            # logs = "NA",
        )


        response, meta = self.model(filled_prompt)
        # response = self.model.generate(
        #     filled_prompt,
        # )

        return AutoMetricLLMOutput(**response) # response

    def results_to_markdown(self, llm_output:AutoMetricLLMOutput):
        md = ""
        for m in llm_output.metrics:
            md += f"**{m.name}**"
            md += f"\n\n- {m.description}"
            md += f"\n\n- Implementation: {m.implementation}"
            if m.code:
                md += f"\n```python\n{m.code}\n```"
            md += "\n\n"
        return md
