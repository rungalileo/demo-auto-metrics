import json


def clean_json_str(json_str: str) -> str:
    """
    Clean the JSON string by removing any newlines, or ticks from the GitHub Flavored Markdown that GPTs like to
    add.

    Even though our instructions specifically ask for JSON without any of the
    GitHub Flavored Markdown backticks, we still get them. This function
    removes them.

    Parameters
    ----------
    json_str : str
        Raw JSON string output from an LLM.

    Returns
    -------
    str
        Cleaned JSON string that can be parsed by the `json` module.
    """
    # Remove any newlines if they exist.
    json_str = json_str.strip()
    json_str = json_str.replace("\n", " ").replace("\r", "")
    # Remove any ticks if they exist.
    json_str = json_str.strip("`").strip()
    # If the text is prefixed with ```json, remove it.
    if json_str.startswith("json"):
        json_str = json_str[4:]
    # And remove any newlines again.
    return json_str.strip()


def parse_response_text(response_text):
    response_text = clean_json_str(response_text)

    return json.loads(response_text)