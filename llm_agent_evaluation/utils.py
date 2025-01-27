import enum
import json
import os
import time
from typing import Literal

from anthropic import Anthropic

import boto3
from botocore.config import Config


AllowedModelNames = Literal[
    'us.anthropic.claude-3-sonnet-20240229-v1:0',
    'us.anthropic.claude-3-haiku-20240307-v1:0',
    'us.anthropic.claude-3-opus-20240229-v1:0'
]


class CustomEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        return isinstance(item, cls) or item in [v.value for v in cls.__members__.values()] 


class BedrockModel:
    """Helper class to invoke LLM from Bedrock"""
    def __init__(self, model_name: AllowedModelNames):
        # `max_tokens_to_sample` can be atmost 4096 in claude-3-sonnet.
        max_tokens_to_sample = 4096

        self.model_name = model_name
        self.inference_params = {
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": 0,
            "top_p": 0.9,
        }

    def _get_client(self) -> boto3.client:
        """Establish `boto3.Session` and retrieve a session client."""
        config = Config(
            read_timeout=300, retries={"max_attempts": 10, "mode": "standard"}
        )
        aws_session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name="us-east-1",
        )
        return aws_session.client("bedrock-runtime", config=config)

    def invoke(self, system: str, prompt: str) -> str:
        """Invoke LLM with the given system and user prompts."""
        client = self._get_client()

        if "anthropic" in self.model_name:
            response = client.invoke_model(
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.inference_params['max_tokens_to_sample'],
                        "system": system,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                        "temperature": self.inference_params['temperature'],
                    }
                ),
            )
            result = json.loads(response["body"].read())["content"][0]["text"]

        if "claude-3-opus" or "claude-3-5-sonnet" in self.model_name:
            time.sleep(30)

        return result


class AnthropicModel:
    """Helper class to invoke LLM from Anthropic"""
    def __init__(self, model_name: str):
        # `max_tokens_to_sample` can be atmost 4096 in claude-3-sonnet.
        max_tokens_to_sample = 4096

        self.model_name = model_name
        self.inference_params = {
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": 0,
            "top_p": 0.9,
        }

    def _get_client(self) -> boto3.client:
        """Retrieve Anthropic client."""
        return Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), max_retries=25)

    def invoke(self, system: str, prompt: str) -> str:
        """Invoke LLM with the given system and user prompts."""
        client = self._get_client()

        response = client.messages.create(
            max_tokens=self.inference_params['max_tokens_to_sample'],
            system=system,
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=self.inference_params['temperature']
        )
        result = response.content[0].text

        # TODO: Fix this
        # if "claude-3-opus" or "claude-3-5-sonnet" in self.model_name:
        if "claude-3-5-sonnet" in self.model_name:
            time.sleep(60)

        return result
