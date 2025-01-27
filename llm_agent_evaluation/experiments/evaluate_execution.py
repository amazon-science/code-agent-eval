"""
Note that these experiments require function-level context.
1. Holistic
    $ python evaluate_execution.py --benchmark swe-bench --type Lite --patch_label 20240617_factory_code_droid --aggregation none

2. Holistic w/ patches
    $ python evaluate_execution.py --benchmark swe-bench --type Lite --patch_label 20240617_factory_code_droid --aggregation none-patch

3. Test-centric
    $ python evaluate_execution.py --benchmark swe-bench --type Lite --patch_label 20240617_factory_code_droid --aggregation test-centric

4. Test-centric w/ patches.
    $ python evaluate_execution.py --benchmark swe-bench --type Lite --patch_label 20240617_factory_code_droid --aggregation test-centric-patch
"""
import argparse
import itertools
import json
import logging
import random
import subprocess
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


load_dotenv()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_all_function_names(source: str) -> List[str]:
    """Extract names of all functions defined in given source code.

    Args:
        source: Function code string.

    Returns:
        Name of the function.
    """
    ast_tree = build_ast_from_source(source)
    function_nodes = list(find_nodes_of_type(ast_tree, 'function_definition'))
    if not function_nodes:
        return None

    function_names = []
    for node in function_nodes:
        for child in node.children:
            if child.type == 'identifier':
                name = child.text.decode('utf-8')
                break
        function_names.append(name)

    return function_names


def extract_flattened_change_patch(chunks: List[Chunk]) -> str:
    """Both `ChangePatch.before_chunks` and `ChangePatch.after_chunks` cache
    all hunks in their respective patches, with or without additional context.

    This helper function concatenates the hunks into a single string. 
    
    While computing lexical scores, if `--full_patch` flag is not provided,
    hunks with no additional context, across the patch are concatenated.

    While computing execution scores, all hunks with function/class-level
    context across the patch are concatenated.
    
    Args:
        chunks: Input list of chunks.

    Returns:
        A string representing the flattened patch.
    """
    all_chunks = {}
    for file_name, grouped_chunks_by_filename in group_chunks_by_attribute(
        chunks, 'filename'
    ).items():
        # Within each file, each is a unique <header-Chunk> pair.
        # We can directly sort by the `start_lineno`
        all_chunks[file_name] = sorted(
            grouped_chunks_by_filename, key=lambda chunk: chunk.start_lineno
        )
    all_chunks = sorted(all_chunks.items(), key=lambda item: item[0])
    all_chunks = list(itertools.chain.from_iterable([item[1] for item in all_chunks]))
    all_chunks = [
        '\n'.join([item[1] for item in chunk.lines]) for chunk in all_chunks
    ]
    return '\n'.join(all_chunks)


def get_test_centric_prompt_inputs_and_ground_truth(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameters for test-centric, execution-specific
    evaluation.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark.
        tests_status_mapper: Maps unit tests to pass/fail status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring
        criterion, and corresponding ground-truth labels.
    """
    instance_id = input_patch.id
    source = extract_flattened_change_patch(
        input_patch.change_patch.after_chunks
    )
    prompt_inputs, true = [], []
    for test_source in gold_patch.test_patch.relevant_tests:
        test_function_name = extract_function_name(test_source)

        # Skip, if a function name can not be retrieved from test source.
        if not test_function_name:
            continue

        # If 'test' does not appear in function name, skip instance.
        if not any(affix in test_function_name for affix in ('test_', '_test')):
            continue

        if benchmark == 'BugsInPy':
            test_label = get_test_label_bugsinpy()
        else:
            test_label = get_test_label_swebench(
                test_function_name,
                instance_id,
                tests_status_mapper,
            )

        if not test_label:
            continue

        prompt_inputs.append(
            ('test-centric', {'source': source, 'test': test_source})
        )
        true.append(test_label)

    return prompt_inputs, true


def get_test_centric_prompt_inputs_and_ground_truth_with_patches(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    patch_label: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameters for test-centric, execution-specific
    evaluation with patches as input.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark.
        patch_label: Patch label.
        tests_status_mapper: Maps unit tests to pass/fail status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring
        criterion, and corresponding ground-truth labels.
    """
    instance_id = input_patch.id

    if not (Path(input_patch.path_to_root) / f'snapshots/{patch_label}').exists():
        return [], []

    patch_text = subprocess.run(
            ["git", "diff", "--function-context"],
            cwd=str(Path(input_patch.path_to_root) / f'snapshots/{patch_label}'),
            check=True,
            capture_output=True,
            text=True,
    ).stdout

    prompt_inputs, true = [], []
    for test_source in gold_patch.test_patch.relevant_tests:
        test_function_name = extract_function_name(test_source)

        # Skip, if a function name can not be retrieved from test source.
        if not test_function_name:
            continue

        # If 'test' does not appear in function name, skip instance.
        if not any(affix in test_function_name for affix in ('test_', '_test')):
            continue

        if benchmark == 'BugsInPy':
            test_label = get_test_label_bugsinpy()
        else:
            test_label = get_test_label_swebench(
                test_function_name,
                instance_id,
                tests_status_mapper,
            )

        if not test_label:
            continue

        prompt_inputs.append(
            ('test-centric-patch', {'patch': patch_text, 'test': test_source})
        )
        true.append(test_label)

    return prompt_inputs, true


def get_function_centric_prompt_inputs_and_ground_truth(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameters for function-centric, execution-specific
    evaluation.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark.
        tests_status_mapper: Maps unit tests to pass/fail status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring
        criterion, and corresponding ground-truth labels.
    """
    instance_id = input_patch.id
    all_tests = gold_patch.test_patch.relevant_tests

    prompt_inputs, true = [], []
    for chunk in input_patch.change_patch.after_chunks:
        source = '\n'.join([item[1] for item in chunk.lines])
        function_names = extract_all_function_names(source)

        if not function_names:
            continue

        relevant_tests, relevant_test_labels = [], []
        for test_source in all_tests:
            # Here, if the given function is `Foo`, we will check whether
            # `Foo` is called in that test function. If not, we skip it.
            relevant = False
            for _name in function_names:
                if _name in test_source:
                    relevant = True
                    break

            if not relevant:
                continue

            test_function_name = extract_function_name(test_source)

            # Skip, if a function name can not be retrieved from test source.
            if not test_function_name:
                continue

            # If 'test' does not appear in function name, skip instance.
            if not any(affix in test_function_name for affix in ('test_', '_test')):
                continue

            if benchmark == 'BugsInPy':
                _test_label = get_test_label_bugsinpy()
            else:
                _test_label = get_test_label_swebench(
                    test_function_name,
                    instance_id,
                    tests_status_mapper,
                )

            if not _test_label:
                continue

            relevant_tests.append(test_source)
            relevant_test_labels.append(_test_label)

        relevant_tests = '\n\n'.join(relevant_tests)
        prompt_inputs.append(
            ('function-centric', {'source': source, 'tests': relevant_tests})
        )

        true.append(
            'pass' if all(value == 'pass' for value in relevant_test_labels)
            else 'fail'
        )

    return prompt_inputs, true


def get_holistic_prompt_inputs_and_ground_truth(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    resolved_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameters for holistic, execution-specific evaluation.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark.
        resolved_status_mapper: Maps instances to build status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring
        criterion, and corresponding ground-truth labels.
    """
    source = extract_flattened_change_patch(
        input_patch.change_patch.after_chunks
    )
    tests = '\n\n'.join(gold_patch.test_patch.relevant_tests)

    prompt_inputs = [
        ('no-aggregation', {'source': source, 'tests': tests})
    ]

    if benchmark == 'BugsInPy':
        raise NotImplementedError
    else:
        if input_patch.id in resolved_status_mapper['resolved']:
            true = ['pass']
        else:
            true = ['fail']

    return prompt_inputs, true


def get_holistic_prompt_inputs_and_ground_truth_with_patches(
    input_patch: Patch,
    gold_patch: Patch,
    benchmark: str,
    patch_label: str,
    resolved_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameters for test-centric, execution-specific
    evaluation with patches as an input.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        benchmark: Name of benchmark.
        patch_label: Patch label.
        resolved_status_mapper: Maps instances to build status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring
        criterion, and corresponding ground-truth labels.
    """
    instance_id = input_patch.id

    patch_text = subprocess.run(
            ["git", "diff", "--function-context"],
            cwd=str(Path(input_patch.path_to_root) / f'snapshots/{patch_label}'),
            check=True,
            capture_output=True,
            text=True,
    ).stdout

    tests = '\n\n'.join(gold_patch.test_patch.relevant_tests)

    prompt_inputs = [
        ('no-aggregation-patch', {'patch': patch_text, 'tests': tests})
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
    aggregation: str,
    filter_by_dependencies: bool,
    benchmark: str,
    patch_label: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
    resolved_status_mapper: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Dict[str, Any]]], List[str]]:
    """Extract prompt names and parameter pairs, and corresponding ground-truth
    for execution-specific evaluation.

    Args:
        input_patch: Input patch.
        gold_patch: Gold patch.
        aggregation: Aggregation strategy for test oracle prediction.
        filter_by_dependencies: Whether to map dependencies between functions
            and tests in function-centric/test-centric evaluation.
        benchmark: Name of benchmark.
        patch_label: Patch label.
        tests_status_mapper: Maps unit tests to pass/fail status.
        resolved_status_mapper: Maps instances to build status.

    Returns:
        Tuple of <prompt_name, prompt_parameters> pairs for a scoring criterion,
        and the corresponding ground truth.
    """
    # If no aggregation, all functions and all tests will be passed to the LLM,
    # predicting whether *all* tests will pass.
    if aggregation == 'none':
        prompt_inputs, true = get_holistic_prompt_inputs_and_ground_truth(
            input_patch,
            gold_patch,
            benchmark,
            resolved_status_mapper,
        )

    elif aggregation == 'none-patch':
        prompt_inputs, true = get_holistic_prompt_inputs_and_ground_truth_with_patches(
            input_patch,
            gold_patch,
            benchmark,
            patch_label,
            resolved_status_mapper,
        )

    # In the case of test-centric aggregation, all/dependent functions will be
    # passed to the LLM, trying to predict whether the *given* test will pass.
    elif aggregation == 'test-centric':
        prompt_inputs, true = get_test_centric_prompt_inputs_and_ground_truth(
            input_patch,
            gold_patch,
            benchmark,
            tests_status_mapper,
        )

    # In the case of test-centric aggregation with patches, all relevant
    # tests will be passed to the LLM along with the entire patch expanded
    # to the function-level context, trying to predict whether all tests
    # associated with this function will pass.
    elif aggregation == 'test-centric-patch':
        prompt_inputs, true = get_test_centric_prompt_inputs_and_ground_truth_with_patches(
            input_patch,
            gold_patch,
            benchmark,
            patch_label,
            tests_status_mapper,
        )

    # In the case of function-centric aggregation, all/dependent tests will
    # be passed to the LLM along with each function, trying to predict
    # whether all/dependent tests associated with this function will pass.
    elif aggregation == 'function-centric':
        prompt_inputs, true = get_function_centric_prompt_inputs_and_ground_truth(
            input_patch,
            gold_patch,
            benchmark,
            tests_status_mapper,
        )

    return prompt_inputs, true


def evaluate_execution(
    input_patch: Patch,
    patch_id_to_gold_patch_mapping: Dict,
    aggregation: str,
    filter_by_dependencies: bool,
    benchmark: str,
    patch_label: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
    resolved_status_mapper: Dict[str, Dict[str, str]],
    model_name: str,
):
    gold_patch = patch_id_to_gold_patch_mapping[input_patch.id]
    prompt_inputs, true = get_prompt_inputs_and_ground_truth(
        input_patch,
        gold_patch,
        args.aggregation,
        args.filter,
        benchmark,
        patch_label,
        tests_status_mapper,
        resolved_status_mapper,
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

    # Logging message for printing LLM predictions.
    msg = f'*** Patch {input_patch.id} ***'
    pairs = zip([output.prediction for output in llm_output], true)
    msg += '\n\t'.join([f"(prediction: {item[0]}, true: {item[1]})" for item in pairs])
    logger.info(msg)

    return (
        input_patch.id,
        [
            {
                'prompt_inputs': prompt_inputs[idx][1],
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
    parser.add_argument('--context_key', type=str, default='function',
                        choices=['none', 'function', 'dependency'],
                        help="Amount of context to include in patches")
    parser.add_argument('--patch_label', type=str,
#                        default='20240509_amazon-q-developer-agent-20240430-dev',
                        default='20240617_factory_code_droid',
                        help="Type of patch to score (e.g., 'gold', 'perturbation', etc.)")
    parser.add_argument('--aggregation', type=str, default='none',
                        choices=['none', 'none-patch', 'test-centric', 'function-centric', 'test-centric-patch'],
                        help="Type of test oracle prediction prompting strategy to use.")
    parser.add_argument('--filter', action='store_true',
                        help="Whether to filter functions and tests in patch based on dependencies")
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
    patches = patch_collector(args.context_key, args.patch_label)

    gold_patches = patch_collector(args.context_key, 'gold')
    patch_id_to_gold_patch_mapping = {patch.id: patch for patch in gold_patches}

    logger.info((
        'Computing execution-specific proxies for diff patches with '
        'function/class-level context.'
    ))

    tests_status_file = f'{benchmark}.{args.patch_label}.test-status.json'
    with open(str(path_to_assets / 'cache' / tests_status_file), 'r') as f:
        tests_status_mapper = json.load(f)

    resolved_status_file = f'{benchmark}.{args.patch_label}.resolved-status.json'
    with open(str(path_to_assets / 'cache' / resolved_status_file), 'r') as f:
        resolved_status_mapper = json.load(f)

    with ThreadPool(16) as pool:
        _outputs = list(
            tqdm(
                pool.imap(
                    partial(
                        evaluate_execution,
                        patch_id_to_gold_patch_mapping=patch_id_to_gold_patch_mapping,
                        aggregation=args.aggregation,
                        filter_by_dependencies=args.filter,
                        benchmark=benchmark,
                        patch_label=args.patch_label,
                        tests_status_mapper=tests_status_mapper,
                        resolved_status_mapper=resolved_status_mapper,
                        model_name=model_name,
                    ),
                    patches,    
                ),
                total=len(patches),
            )
        )

    outputs = {_output[0]: _output[1] for _output in _outputs if _output}
    filename = f'{benchmark}.{args.patch_label}.execution.{args.aggregation}.json'
    Path(path_to_assets / f'results-{args.model_name}').mkdir(parents=True, exist_ok=True)
    with open(str(path_to_assets / f'results-{args.model_name}' / filename), 'w') as f:
        json.dump(outputs, f, indent=2)
