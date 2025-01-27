import pathlib
from typing import Any, Dict, List

import boto3
from botocore.config import Config

from llm_agent_evaluation.data.patch_utils import Chunk, Patch
from llm_agent_evaluation.utils import (
    AllowedModelNames,
    BedrockModel,
    AnthropicModel,
)
from llm_agent_evaluation.scorers import BaseScorer
from llm_agent_evaluation.scorers.llm_utils import (
    extract_tag,
    PromptFactory,
    LLMCriticOutput,
)


class LLMScorer(BaseScorer):
    """An LLM-based critic that scores a patch based on specified criteria."""
    _name = 'llm'

    def __init__(
        self,
        benchmark: str,
        to_path: pathlib.Path,
        model_name: AllowedModelNames,
    ):
        """Initialize LLM-based critic.

        Args:
            benchmark: Benchmark dataset name.
            to_path: Path to dataset resources.
            model_name: LLM model name.
        """
        self.logger = self.init_scoring_logger(benchmark, to_path)
        self.logger.info(f'LLM-based evaluation of {benchmark} benchmark.')

        # LLM related objects
        if model_name.startswith('anthropic') or model_name.startswith('us.anthropic'):
            self.model = BedrockModel(model_name=model_name)
        else:
            self.model = AnthropicModel(model_name=model_name)
        self.prompt_factory = PromptFactory()

    def score(self, prompt_inputs: List[Dict[str, Any]]) -> List[Any]:
        """LLM-based scoring for a single patch.

        Args:
            prompt_inputs: List of <prompt-name, prompt-parameters> pairs
                for a given input patch.

        Returns:
            A list of predictions and analyses, one per prompt input.
        """
        system = self.prompt_factory.get_prompt('system', {})

        llm_preds = []
        for prompt_name, prompt_params in prompt_inputs:
            response = self.model.invoke(
                system=system,
                prompt=self.prompt_factory.get_prompt(prompt_name, prompt_params)
            )
            llm_preds.append(LLMCriticOutput(
                prediction=extract_tag('prediction', response, logger=self.logger),
                confidence=float(extract_tag('confidence', response, logger=self.logger)),
                analysis=extract_tag('analysis', response, logger=self.logger),
            ))

        return llm_preds
