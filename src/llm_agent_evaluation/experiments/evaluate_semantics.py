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

from llm_agent_evaluation.external.ast.explorer import (
    build_ast_from_source,
    find_nodes_of_type,
)
from llm_agent_evaluation.scorers.llm import LLMScorer
from llm_agent_evaluation.data.patch_utils import (
    Chunk,
    Patch,
    group_chunks_by_attribute,
)

from utils import *
from unittest.mock import patch


load_dotenv()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_micro_prompt_inputs_and_ground_truth(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameter pairs, and corresponding ground-truth
    for hunk-level (with function-level context) semantics-specific evaluation.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark dataset.
        tests_status_mapper: Maps unit tests to pass/fail status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring criterion,
        and the corresponding ground truth.
    """
    prompt_inputs, true = [], []
    for chunk in input_patch.change_patch.after_chunks:
        # Skip if hunk context is a class.
        if 'class ' in chunk.header:
            continue

        function1 = '\n'.join([item[1] for item in chunk.lines])
        function1_name = extract_function_name(function1)

        # Find corresponding chunk in gold patch.
        equivalent_chunk = None
        for gold_chunk in gold_patch.change_patch.after_chunks:
            if 'class ' in chunk.header:
                continue

            function2 = '\n'.join([item[1] for item in chunk.lines])
            function2_name = extract_function_name(function2)

            if function1_name == function2_name:
                equivalent_chunk = gold_chunk
                break

        # If corresponding location was not changed in the gold patch, skip...
        if not equivalent_chunk:
            continue

        equivalent_test_name = None
        for test_source in gold_patch.test_patch.relevant_tests:
            test_function_name = extract_function_name(test_source)

            # Skip, if a function name can not be retrieved from test source.
            if not test_function_name:
                continue

            # Finding equivalent test for Foo by checking for test_Foo or Foo_test.
            if test_function_name.startswith('test_'):
                if test_function_name[5:] == function1_name:
                    equivalent_test_name = test_function_name
                    break
            elif test_function_name.endswith('_test'):
                if test_function_name[:-5] == function1_name:
                    equivalent_test_name = test_function_name
                    break

        # If we do not find equivalent test for given function, skip...
        if not equivalent_test_name:
            continue

        if benchmark == 'BugsInPy':
            test_label = get_test_label_bugsinpy()
        else:
            test_label = get_test_label_swebench(
                equivalent_test_name,
                instance_id,
                tests_status_mapper,
            )

        if not test_label:
            continue

        prompt_inputs.append(
            ('semantic-equivalence', {'snippet1': function1, 'snippet2': function2})
        )
        true.append(test_label)

    return prompt_inputs, true


def get_macro_prompt_inputs_and_ground_truth(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    resolved_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameter pairs, and corresponding ground-truth
    for patch-level semantics-specific evaluation.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark dataset.
        resolved_status_mapper: Maps instances to build status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring criterion,
        and the corresponding ground truth.
    """
    prompt_inputs = [
        ('patch-equivalence', {
            'patch1': input_patch.change_patch.text,
            'patch2': gold_patch.change_patch.text,
        })
    ]

    if benchmark == 'BugsInPy':
        raise NotImplementedError
    else:
        if input_patch.id in resolved_status_mapper['resolved']:
            true = ['pass']
        else:
            true = ['fail']

    return prompt_inputs, true


def get_macro_prompt_inputs_and_ground_truth_with_git(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    resolved_status_mapper: Dict[str, Dict[str, str]],
    patch_label: str,
    additional_context: str,
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameter pairs, and corresponding ground-truth
    for patch-level semantics-specific evaluation using `git diff` utilities.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark dataset.
        resolved_status_mapper: Maps instances to build status.
        patch_label: Patch label
        additional_context: Additional context to be included in the unified diff.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring criterion,
        and the corresponding ground truth.
    """
    if not additional_context:
        diff_args = ["git", "diff"]
    else:
        diff_args = ["git", "diff", additional_context]

    # Get git diff with additional context lines
    patch1 = subprocess.run(
            diff_args,
            cwd=str(Path(input_patch.path_to_root) / f'snapshots/{patch_label}'),
            check=True,
            capture_output=True,
            text=True,
    ).stdout

    patch2 = subprocess.run(
            diff_args,
            cwd=str(Path(input_patch.path_to_root) / 'snapshots/gold'),
            check=True,
            capture_output=True,
            text=True,
    ).stdout

    prompt_inputs = [
        ('patch-equivalence', {'patch1': patch1, 'patch2': patch2})
    ]

    if benchmark == 'BugsInPy':
        raise NotImplementedError
    else:
        if input_patch.id in resolved_status_mapper['resolved']:
            true = ['pass']
        else:
            true = ['fail']

    return prompt_inputs, true


def get_prompt_inputs_and_ground_truth(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
    resolved_status_mapper: Dict[str, Dict[str, str]],
    micro_eval: bool,
    patch_label: str,
    additional_context: str,
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameter pairs, and corresponding ground-truth
    for semantics-specific evaluation.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark dataset.
        tests_status_mapper: Maps unit tests to pass/fail status.
        resolved_status_mapper: Maps instances to build status.
        micro_eval: If True, functional equivalence is evaluated. Otherwise,
            patch equivalence is evaluated.
        patch_label: Patch label.
        additional_context: Additional context to be included in the unified diff.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring criterion,
        and the corresponding ground truth.
    """
    if micro_eval:
        return get_micro_prompt_inputs_and_ground_truth(
            input_patch,
            gold_patch,
            benchmark,
            tests_status_mapper,
        )
    else:
        return get_macro_prompt_inputs_and_ground_truth_with_git(
            input_patch,
            gold_patch,
            benchmark,
            resolved_status_mapper,
            patch_label,
            additional_context,
        )


def evaluate_semantics(
    input_patch: Patch,
    patch_id_to_gold_patch_mapping: Dict,
    benchmark: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
    resolved_status_mapper: Dict[str, Dict[str, str]],
    micro_eval: bool,
    patch_label: str,
    additional_context: str,
    model_name: str,
):
    """Evaluate semantics of a given input patch against a gold patch.

    Args:
        input_patch: Input patch.
        patch_id_to_gold_patch_mapping: Mapping from input patch ID to gold patch ID.
        benchmark: Name of benchmark dataset.
        tests_status_mapper: Maps unit tests to pass/fail status.
        resolved_status_mapper: Maps instances to build status.
        micro_eval: If True, functional equivalence is evaluated. Otherwise,
            patch equivalence is evaluated.
        patch_label: Patch label.
        additional_context: Amount of additional context in diff.
        model_name: Claude model name in Bedrock.
    """
    gold_patch = patch_id_to_gold_patch_mapping[input_patch.id]
    prompt_inputs, true = get_prompt_inputs_and_ground_truth(
        input_patch,
        gold_patch,
        benchmark,
        tests_status_mapper,
        resolved_status_mapper,
        micro_eval,
        patch_label,
        additional_context,
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
    parser.add_argument('--micro_eval', action='store_true',
                        help=("At a micro-level, functional equivalence is evaluated,"
                              "comparing hunks expanded to function-level context."
                              "At a macro-level, patch equivalence is evaluated."))
    parser.add_argument('--additional_context', type=str,
                        default='-U10', choices=['none', 'U10', 'function'],
                        help="Amount of additional context to include")
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

    gold_patches = patch_collector('function', 'gold')
    patch_id_to_gold_patch_mapping = {patch.id: patch for patch in gold_patches}

    if args.additional_context == 'none':
        additional_context = ''
    elif args.additional_context == 'U10':
        additional_context = '-U10'
    elif args.additional_context == 'function':
        additional_context = '--function-context'

    logger.info('Computing semantics-specific proxies for diff patches.')

    tests_status_file = f'{benchmark}.{args.patch_label}.test-status.json'
    with open(str(path_to_assets / 'cache' / tests_status_file), 'r') as f:
        tests_status_mapper = json.load(f)

    resolved_status_file = f'{benchmark}.{args.patch_label}.resolved-status.json'
    with open(str(path_to_assets / 'cache' / resolved_status_file), 'r') as f:
        resolved_status_mapper = json.load(f)

    with ThreadPool(4) as pool:
        _outputs = list(
            tqdm(
                pool.imap(
                    partial(
                        evaluate_semantics,
                        patch_id_to_gold_patch_mapping=patch_id_to_gold_patch_mapping,
                        benchmark=benchmark,
                        tests_status_mapper=tests_status_mapper,
                        resolved_status_mapper=resolved_status_mapper,
                        micro_eval=args.micro_eval,
                        patch_label=args.patch_label,
                        additional_context=additional_context,
                        model_name=model_name,
                    ),
                    patches,    
                ),
                total=len(patches),
            )
        )

    outputs = {_output[0]: _output[1] for _output in _outputs if _output}
    suffix = 'functional' if args.micro_eval else 'patch'
    filename = f'{benchmark}.{args.patch_label}.semantics.{args.additional_context}.{suffix}.json'
    Path(path_to_assets / f'results-{args.model_name}').mkdir(parents=True, exist_ok=True)
    with open(str(path_to_assets / f'results-{args.model_name}' / filename), 'w') as f:
        json.dump(outputs, f, indent=2)
