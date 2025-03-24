from pydantic import BaseModel


class LLMOutputSchema(BaseModel):
    """Custom base class for all schema models."""
    pass

class LLMOutputMetric(LLMOutputSchema):
    name: str
    description: str
    implementation: str
    code: str

class AutoMetricLLMOutput(LLMOutputSchema):
    metrics: list[LLMOutputMetric]


CODE_INSTRUCTIONS_PROMPT = """
The code implementations must be in a specific format to be integrated into a unique platform. Below are instructions for how you should implement a custom metric in python.
You must follow these instructions for each code snippet.

For example, here is a snippet to create a custom metric that measured the length of the response:
1. Define a `scorer_fn`. The scorer function is provided the row-wise inputs and is expected to generate outputs for each response. The expected signature for this function is:
```
def scorer_fn(*, index: Union[int, str], node_input: str, node_output: str, **kwargs: Any) -> Union[float, int, bool, str, None]:
  ...
```
`node_name`, `node_type`, `node_id` and `tools` are all specific to workflows/multi step chains.
`dataset_variables` contains key-value pairs of variables that are passed in from the dataset in prompt evaluation runs, but can also be used to get the target/ground truth in multi step runs.
`index` is the index of the row in the dataset, `node_input` is the input to the node, and `node_output` is the output from the node.

The index parameter is the index of the row in the dataset, node_input is the input to the node, and node_output is the output from the node.

2. Define `aggregator_fn`. The aggregator function takes in an array of the row-wise outputs from your scorer and allows you to generate aggregates from those.
The expected signature for the aggregator function is:
```
def aggregator_fn(*, scores: List[Union[float, int, bool, str, None]]) -> Dict[str, Union[float, int, bool, str, None]]:
        ...
```

3. (Optional, but recommended) `score_type`: The scorer_type function is used to define the Type of the score that your scorer generates.
The expected signature for this function is:
```
def score_type() -> Type[float] | Type[int] | Type[str] | Type[bool]:
    ...
```
Note that the return type is a `Type` object like `float`, not the actual type itself.
Defining this function is necessary for sorting and filtering by scores to work correctly.
If you don't define this function, the scorer is assumed to generate float scores by default.

4. (Optional) scoreable_node_types_fn: If you want to restrict your scorer to only run on specific node types, you can define this function which returns a list of node types that your scorer should run on. The expected signature for this function is:
```
def scoreable_node_types_fn() -> List[str]:
        ...
```
If you don't define this function, your scorer will run on `llm` and `chat` nodes by default.

5. Example. For example, let's say we wanted to create a custom metric that measured the length of the response.
We would define a scorer_fn function, and an aggregator_fn function. Here is how the code snipped will look like:
```
from typing import List, Dict, Type


def scorer_fn(*, response: str, **kwargs) -> int:
    return len(response)


def aggregator_fn(*, scores: List[str]) -> Dict[str, int]:
    return {
        "Total Response Length": sum(scores),
        "Average Response Length": sum(scores) / len(scores),
    }

def score_type() -> Type:
    return int

def scoreable_node_types_fn() -> List[str]:
    return ["llm", "chat"]
```

"""

AVAILABLE_METRICS_PROMPT = """
PRE-CONFIGURED METRICS:

Name: Input PII
Description: Tracks the presence of personal identifiable information in the user's input.
Implementation: model

Name: Input Sexism
Description: Measures how 'sexist' a user's input might be perceived ranging in the values of 0-1 (1 being more sexist).
Implementation: model

Name: Input Tone
Description: Classifies the sentiment of the user's input into one of joy, love, fear, surprise, sadness, anger, annoyance, confusion or neutral.
Implementation: model

Name: Input Toxicity
Description: Measures the presence and severity of harmful, offensive, or abusive language.
Implementation: model

Name: Output PII
Description: Tracks the presence of personal identifiable information in the LLM's responses.
Implementation: model

Name: Prompt Injection
Description: Detects and classifies prompt injection attacks.
Implementation: model

Name: Output Tone
Description: Detects and classifies prompt injection attacks.
Implementation: model

Name: Output Toxicity
Description: Measures the presence and severity of harmful, offensive, or abusive language in the model's response.
Implementation: model

Name: Uncertainty
Description: A measure of the model's own confusion in its output. Higher scores indicate higher uncertainty.
Implementation: model

-- RAG METRICS --
Name: RAG Completeness
Description: Measures how thoroughly your model's response covered the relevant information available in the context provided
Implementation: model

Name: RAG Context Adherence
Description: Measures whether the LLM's response is supported by (or baked in) the context provided.
Implementation: model

Name: RAG Chunk Attribution & Utilization
Description: For each chunk retrieved in a RAG pipeline, attribution measures whether or not that chunk had an effect on the model's response and utilization measures the fraction of the text in that chunk that had an impact on the model's response.
Implementation: model

Name: Instruction Adherence
Description: Measures whether the LLM is adhering to its system or prompt instructions.
Implementation: model

-- AGENT METRICS --
Name: Action Advancement
Description: Detects whether a user successfully accomplished or advanced towards their goal in a single turn interaction.
Implementation: model

Name: Action Completion
Description: Detects whether the user successfully accomplished all of their goals in a multiturn interaction.
Implementation: model

Name: Tool Selection Quality
Description: Detects whether the model selected the right tools with the right arguments.
Implementation: model

Name: Tool Error Rate
Description: Detects whether the Tool executed successfully (i.e. without errors).
Implementation: model


[END PRE-CONFIGURED METRICS]
"""

AUTO_METRICS_PROMPT = """
You are an AI assistant tasked with suggesting metrics for users of a platform that enables logging and analysis of LLM app inputs and outputs. 
Based on a brief description of a user's app, you will suggest a few relevant metrics that could help evaluate and improve the app's performance.

APPLICATION DESCRIPTION:
{user_app_description}

The users has also specified that the following labels apply to their app:
{app_labels}

Your task is to suggest 5-7 metrics that would be most useful for evaluating and improving this specific LLM app. Below you will also find a list of pre-configured metrics that you may use in your response.
Use the list of pre-configured metrics only as a reference, not as a single source of possible metrics. 
Suggest up to 5 most relevant pre-configured metrics. The rest should be novel metrics to evaluate.

Requirements:
1. Metrics should be relevant to the app's purpose and functionality.
2. Metrics should be measurable provided the prompt, user input, and LLM response. Do not suggest metrics that require additional rating/annotation from a human.
3. Select ONLY those pre-configured metrics that are relevant to the application in question and would be useful in evaluating this specific application. For example, DO NOT propose RAG metrics for a
non-RAG use case. Similarly, do not propose agentic metrics for non-agentic use cases. Do not propose PII if PII is not expected in user input/LLM output.
5. The metrics should be specific. When specific instructions are detailed in the Application Description, generate metrics to measure how well the LLM output adheres to these specific requirements.
6. Suggest metrics that could provide actionable insights for improvement.
7. Do not suggest overly broad, subjective custom metrics. "Response Correctness", or "Recommendation Quality" are examples of broad metrics that may be hard to define and evaluate.

When considering metrics, think about various aspects of the app, such as:
- Input quality and diversity
- Output accuracy and relevance
- Response time and efficiency
- User engagement and satisfaction
- Error rates and types, specific to the application in question
- Task completion rates

{available_metrics}

Respond with a list of JSON objects matching this schema. Do not add any extra text before or after the JSON:
{{
  "metrics": [
    {{
      "name": string,
      "description": string,
      "implementation": string,
      "code": string
    }}
  ]
}}

"metrics" is an array of the metrics you come up with.
- "name" is the name of the metric.
- "description" is a short description of the metric in natural language. Make the description specific to the client's application.
- "implementation": decide whether or not this metric can be implemented in code or if it needs a model to approximate it.
For example simple metrics like counting words, comparing agains facts, can be implemented with code. Others will prompt an LLM. \
If this is a pre-configured mettic, respond with "galileo". Respond with one of: ["code", "model", "galileo"].
- "code": if you selected "code" for the implementation, generate a python code snippet that can be used to approximate this metric. Otherwise, return an empty string.
Refer to instructions below on how to format the code. Provide complete implementations - do not leave placeholder functions for the user to fill in.
If you can't provide a full implementation, then default to implementation=model.
- be careful not to output duplicate metrics


INSTRUCTIONS FOR CODE SNIPPET
{code_instructions}
"""

def get_filled_prompt(
    user_app_description, 
    app_labels,
):
    return AUTO_METRICS_PROMPT.format(
        user_app_description=user_app_description,
        app_labels=app_labels,
        available_metrics=AVAILABLE_METRICS_PROMPT,
        code_instructions=CODE_INSTRUCTIONS_PROMPT
    )