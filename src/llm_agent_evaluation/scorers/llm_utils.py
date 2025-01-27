import json
import logging
import os
import re
import time
import typing
import yaml
from dataclasses import dataclass
from os.path import abspath, dirname
from pathlib import Path
from typing import List, Dict, Optional

PATH_TO_PROMPT = Path(dirname(abspath(__file__))) / "prompt_templates"


@dataclass
class Prompt:
    name: str
    parameters: list[str]
    text: str


@dataclass
class LLMCriticOutput:
    prediction: str
    confidence: float
    analysis: str


class PromptFactory:
    """Utility to load get prompts from `prompt_templates`."""
    def __init__(self):
        self.prompts: Dict[str, Prompt] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompts from prompt files."""
        for prompt_file in os.listdir(PATH_TO_PROMPT):
            with open(PATH_TO_PROMPT / prompt_file, 'r') as file:
                data = yaml.safe_load(file)
                for prompt_data in data.get('prompts', []):
                    prompt = Prompt(**prompt_data)
                    self.prompts[prompt.name] = prompt

    def get_prompt(
        self,
        prompt_name: str,
        prompt_parameters: Dict[str, str]
    ) -> str:
        """Get the completed prompt text.
        
        Args:
            prompt_name: Prompt identifier.
            prompt_parameters: A mapping of parameter names to values.

        Returns:
            The completed prompt text.
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        prompt: Prompt = self.prompts[prompt_name]
        completed_prompt: str = prompt.text

        for param in prompt.parameters:
            if param not in prompt_parameters:
                raise ValueError(
                    f"Prompt '{prompt_name}' requires parameter '{param}'")

            completed_prompt = completed_prompt.replace(
                '{' + param + '}', str(prompt_parameters[param]))

        return completed_prompt


def extract_tag_list(
    tag: str,
    text: str,
    remove_leading_newline: bool = False,
    logger: logging.Logger = None,
) -> List[str]:
    """Extract a list of tags from a given XML string.

    Args:
        tag: The XML tag to extract.
        text: The XML string.
        remove_leading_newline: Whether to remove the leading newline from the
            extracted values.
        logger: The logger to use for warnings.

    Returns:
        A list of values extracted from the provided tag.
    """
    # Define a regular expression pattern to match the tag
    pattern = rf"<{tag}(?:\s+[^>]*)?>(.*?)</{tag}>"

    # Use re.findall to extract all occurrences of the tag
    values = re.findall(pattern, text, re.DOTALL)

    if len(values) == 0:
        pattern = rf"<{tag}(?:\s+[^>]*)?>(.*)"
        values = re.findall(pattern, text, re.DOTALL)
        if len(values) > 0:
            logger.warning(f"'{tag}' tag was found, but had no closing tag.")
        else:
            logger.warning(f"'{tag}' tag was not found.")

    if remove_leading_newline:
        values = [v[1:] if v[0] == "\n" else v for v in values]
    return values


def extract_tag(
    tag: str,
    text: str,
    remove_leading_newline: bool = False,
    logger: logging.Logger = None,
) -> str:
    """Extract a tag from a given XML string.

    Args:
        tag: The XML tag to extract.
        text: The XML string.
        remove_leading_newline: Whether to remove the leading newline from the
            extracted values.
        logger: The logger to use for warnings.

    Returns:
        An extracted string.
    """
    values = extract_tag_list(tag, text, remove_leading_newline, logger)
    if len(values) > 0:
        return values[0]
    else:
        return ""
