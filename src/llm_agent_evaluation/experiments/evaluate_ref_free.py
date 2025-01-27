import argparse
import itertools
import json
import logging
import random
import subprocess
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv

from tqdm import tqdm

from llm_agent_evaluation.data.patch_utils import Patch
from llm_agent_evaluation.scorers.llm import LLMScorer

from utils import *


load_dotenv()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_prompt_inputs_and_ground_truth(
    input_patch: Patch,
    benchmark: str,
    resolved_status_mapper: Dict[str, Dict[str, str]],
    patch_label: str,
    additional_context: str,
    use_hints: bool
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameter pairs, and corresponding ground-truth
    for reference-free evaluation.

    Args:
        input_patch: Input patch.
        benchmark: Name of benchmark dataset.
        resolved_status_mapper: Maps instances to build status.
        patch_label: Patch label.
        additional_context: Additional context to be included in the unified diff.
        use_hints: If True, uses problem hints.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring criterion,
        and the corresponding ground truth.
    """
    path_to_root = Path(input_patch.path_to_root)
    with open(path_to_root / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    if additional_context == 'none':
        diff_args = ["git", "diff"]
    elif additional_context == 'U10':
        diff_args = ["git", "diff", '-U10']
    elif additional_context == 'function':
        diff_args = ["git", "diff", "--function-context"]

    # Get git diff with additional context lines
    patch = subprocess.run(
            diff_args,
            cwd=str(Path(input_patch.path_to_root) / f'snapshots/{patch_label}'),
            check=True,
            capture_output=True,
            text=True,
    ).stdout

    if use_hints:
        prompt_inputs = [
            ('patch-analysis-with-hints', {
                'issue_description': metadata['problem_statement'],
                'hints': metadata['hints_text'],
                'patch': patch,
                }
            )
        ]
    else:
        prompt_inputs = [
            ('patch-analysis-without-hints', {
                'issue_description': metadata['problem_statement'],
                'patch': patch,
                }
            )
        ]

    if benchmark == 'BugsInPy':
        raise NotImplementedError
    else:
        if input_patch.id in resolved_status_mapper['resolved']:
            true = ['pass']
        else:
            true = ['fail']

    return prompt_inputs, true


def evaluate_without_reference(
    input_patch: Patch,
    benchmark: str,
    resolved_status_mapper: Dict[str, Dict[str, str]],
    patch_label: str,
    additional_context: str,
    use_hints: bool,
    model_name: str,
):
    """Evaluate semantics of a given input patch against a gold patch.

    Args:
        input_patch: Input patch.
        benchmark: Name of benchmark dataset.
        resolved_status_mapper: Maps instances to build status.
        patch_label: Patch label.
        additional_context: Amount of additional context in diff.
        use_hints: If True, use hints for evaluation.
        model_name: Claude model name in Bedrock.
    """
    prompt_inputs, true = get_prompt_inputs_and_ground_truth(
        input_patch,
        benchmark,
        resolved_status_mapper,
        patch_label,
        additional_context,
        use_hints,
    )

    if not prompt_inputs or not true:
        return None

    # Initialize LLM-based evaluation proxy scorer.
    llm_scorer = LLMScorer(
        benchmark=benchmark,
        to_path=path_to_assets,
        model_name=model_name,
    )

    # Get LLM prediction.
    llm_output = llm_scorer.score(prompt_inputs)

    msg = f'*** Patch {input_patch.id} ***'
    pairs = zip([output.prediction for output in llm_output], true)
    msg += '\n\t'.join([f"(prediction: {item[0]}, true: {item[1]})" for item in pairs])
    logger.info(msg)

    return (
        input_patch.id,
        [
            {
                'prompt_inputs': prompt_inputs[idx],
                'true': true[idx],
                'pred': output.prediction,
                'confidence': output.confidence,
                'analysis': output.analysis,
            } for idx, output in enumerate(llm_output)
        ],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM-Based Execution-Free Evalutation of Agent Workflows')

    ## Pipeline arguments
    parser.add_argument('--path_to_assets', type=str, default='../resources',
                        help="Path to dataset resources.")
    parser.add_argument('--benchmark', type=str, default='swe-bench',
                        choices=['BugsInPy', 'swe-bench'], help='Benchmark dataset')
    parser.add_argument('--type', type=str, default='Lite',
                        choices=['Lite', 'test'], help='SWE-Bench variant')
    parser.add_argument('--patch_label', type=str,
#                        default='20240509_amazon-q-developer-agent-20240430-dev',
                        default='20240617_factory_code_droid',
                        help="Type of patch to score (e.g., 'gold', 'perturbation', etc.)")
    parser.add_argument('--additional_context', type=str,
                        default='-U10', choices=['none', 'U10', 'function'],
                        help="Amount of additional context to include")
    parser.add_argument('--use_hints', action='store_true',
                        help=("If True, use hints for problem evaluation."))
    parser.add_argument('--model_name', default='sonnet-3-5',
                        choices=['opus-3', 'sonnet-3', 'sonnet-3-5'], help='Claude model key')
    parser.add_argument('--use_api', action='store_true',
                        help=("If True, uses Anthropic API, else Bedrock"))

    args = parser.parse_args()
    path_to_assets = Path(args.path_to_assets).resolve()

    if args.model_name == 'sonnet-3':
        if args.use_api:
            raise ValueError("Bedrock can be used without rate limits for claude-sonnet-3.")
        else:
            model_name = SONNET_3

    elif args.model_name == 'opus-3':
        if args.use_api:
            model_name = ANTHROPIC_OPUS_3
        else:
            model_name = OPUS_3

    elif args.model_name == 'sonnet-3-5':
        if args.use_api:
            model_name = ANTHROPIC_SONNET_3_5
        else:
            model_name = SONNET_3_5

    if args.benchmark == 'BugsInPy':
        from llm_agent_evaluation.data.bugsinpy.collect import BugsInPyPatchCollector
        benchmark = args.benchmark
        PatchCollectorCls = BugsInPyPatchCollector
        _args = [path_to_assets]
    elif args.benchmark == 'swe-bench':
        benchmark = f'swe-bench_{args.type}' if args.type == 'Lite' else args.benchmark
        from llm_agent_evaluation.data.swe_bench.collect import SWEBenchPatchCollector
        PatchCollectorCls = SWEBenchPatchCollector
        _args = [path_to_assets, args.type]

    patch_collector = PatchCollectorCls(*_args)
    patches = patch_collector('function', args.patch_label)

    logger.info('Computing reference-free proxies for diff patches.')

    resolved_status_file = f'{benchmark}.{args.patch_label}.resolved-status.json'
    with open(str(path_to_assets / 'cache' / resolved_status_file), 'r') as f:
        resolved_status_mapper = json.load(f)

    with ThreadPool(16) as pool:
        _outputs = list(
            tqdm(
                pool.imap(
                    partial(
                        evaluate_without_reference,
                        benchmark=benchmark,
                        resolved_status_mapper=resolved_status_mapper,
                        patch_label=args.patch_label,
                        additional_context=args.additional_context,
                        use_hints=args.use_hints,
                        model_name=model_name,
                    ),
                    patches,    
                ),
                total=len(patches),
            )
        )

    outputs = {_output[0]: _output[1] for _output in _outputs if _output}
    suffix = 'patch.hints' if args.use_hints else 'patch'
    filename = f'{benchmark}.{args.patch_label}.ref-free.{args.additional_context}.{suffix}.json'
    Path(path_to_assets / f'results-{args.model_name}').mkdir(parents=True, exist_ok=True)
    with open(str(path_to_assets / f'results-{args.model_name}' / filename), 'w') as f:
        json.dump(outputs, f, indent=2)
