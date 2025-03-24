import time
import asyncio
from functools import wraps
import openai
import anthropic
from typing import Any, Callable, Coroutine, TypeVar, Union
from typing_extensions import ParamSpec

IdType = Union[str, int]

P = ParamSpec("P")
T = TypeVar("T")


from src.llm_tools.response_parsing import parse_response_text

class AnthropicError(Exception):
    pass

def async_to_sync(fn: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """Decorate an async function to convert it to an ordinary function.

    Calling the decorator function will run the original function in a new event loop,
    and return the result when the loop completes.
    """

    @wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        return asyncio.run(fn(*args, **kwargs))

    return inner


def calc_cost(usage, price_p, price_p_cached, price_r, price_p_cache_write=0):
    if hasattr(usage, 'cache_read_input_tokens'):
        pc = usage.cache_read_input_tokens
        pcw = usage.cache_creation_input_tokens
        pn = usage.input_tokens
        r = usage.output_tokens
    else:
        pc = usage.prompt_tokens_details.cached_tokens
        pn = usage.prompt_tokens - pc
        r = usage.completion_tokens
        pcw = 0

    return price_p * pn + price_p_cached * pc + price_r * r + price_p_cache_write * pcw


def calc_cost_o3_mini(usage):
    price_p, price_p_cached, price_r = (
        1.10 / 1_000_000,
        0.55 / 1_000_000,
        4.40 / 1_000_000,
    )
    return calc_cost(usage, price_p, price_p_cached, price_r)


def calc_cost_claude_3_7_sonnet(usage):
    price_p, price_p_cached, price_r, price_p_cache_write = (
        3 / 1_000_000,
        0.3 / 1_000_000,
        15 / 1_000_000,
        3.75 / 1_000_000,
    )
    return calc_cost(usage, price_p, price_p_cached, price_r, price_p_cache_write)


def call_o3_mini(
    messages,
    api_key,

    use_structured_output=False,
    structured_output_model=None,
    max_completion_tokens=None,
):
    if not api_key:
        raise ValueError("No API key provided")

    client = openai.OpenAI(api_key=api_key)
    
    t1=time.time()

    response_format_kwargs = {}
    
    request_fn = client.chat.completions.create
    if use_structured_output:
        request_fn = client.beta.chat.completions.parse
        response_format_kwargs['response_format'] = structured_output_model

    r = request_fn(
        model='o3-mini',
        messages=messages,

        max_completion_tokens=max_completion_tokens,

        **response_format_kwargs,
    )

    latency = time.time() - t1

    cost = calc_cost_o3_mini(r.usage)

    reasoning_tokens = r.usage.completion_tokens_details.reasoning_tokens
    output_tokens = r.usage.completion_tokens - reasoning_tokens

    response_text = r.choices[0].message.content
    finish_reason = r.choices[0].finish_reason
    
    meta = {
        'latency': latency,

        'cost': cost,

        'token_info': {
            'total_tokens': r.usage.completion_tokens,
            'reasoning_tokens': reasoning_tokens, 
            'output_tokens': output_tokens,
        },

        'finish_reason': finish_reason,
    }

    return response_text, meta


def call_claude_3_7_sonnet(
    messages,
    client,

    claude_3_7_reasoning_budget=4_000,
    claude_3_7_output_budget=2_000,
):
    t1 = time.time()
    
    r = client.messages.create(
        model='claude-3-7-sonnet-20250219',
        messages=messages,
        thinking={
            "type": "enabled",
            "budget_tokens": claude_3_7_reasoning_budget,
        },

        max_tokens = claude_3_7_reasoning_budget + claude_3_7_output_budget,
    )

    total_tokens = client.messages.count_tokens(
        model='claude-3-7-sonnet-20250219',
        messages=[
            {
                'role': 'assistant', 
                'content': r.content
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": claude_3_7_reasoning_budget,
        },
    
    )
    total_tokens = total_tokens.input_tokens
    
    final_tokens = client.messages.count_tokens(
        model='claude-3-7-sonnet-20250219',
        messages=[
            {
                'role': 'assistant', 
                'content': [bl for bl in r.content if bl.type not in ["thinking", "redacted_thinking"]]
            }
        ],
    )
    final_tokens = final_tokens.input_tokens
    thinking_tokens = total_tokens - final_tokens

    latency = time.time() - t1

    cost = calc_cost_claude_3_7_sonnet(r.usage)


    return r.content[-1].text, {
        'latency': latency,

        'cost': cost,

        'token_info': {
            'total_tokens': total_tokens,
            'reasoning_tokens': thinking_tokens,
            'output_tokens': final_tokens,
        },

        'finish_reason': r.stop_reason,

        'usage': r.usage.model_dump(),
    }


def call_claude_3_7_sonnet_and_parse_with_retries(
    messages,
    client,
    max_tries=3,
    claude_3_7_reasoning_budget=4_000,
    claude_3_7_output_budget=2_000,

    request_id=None,
):
    for try_idx in range(max_tries):
        response, meta = call_claude_3_7_sonnet(
            messages, 
            client, 
            claude_3_7_reasoning_budget=claude_3_7_reasoning_budget, 
            claude_3_7_output_budget=claude_3_7_output_budget
        )


        try:
            return parse_response_text(response), meta
        except Exception as e:
            stop_reason = meta.get('finish_reason', None)
            print(f"{request_id}: Error parsing response (try {try_idx+1}/{max_tries}): stop_reason={stop_reason}, error={e}")
            continue
    return response, meta
    # raise AnthropicError(f"{request_id}: Failed to parse model response {max_tries}/{max_tries} times")
