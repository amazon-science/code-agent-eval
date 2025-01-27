"""Lexical proxies are reference-free metrics. Here, we compute:
    (1) edit distance
    (2) cosine similarity, for all patches.

Configurations:
1. No context: All hunks within the predicted change patch are grouped
   based on the headers, and proxies are computed for all groups.
2. Function context: All hunks within the predicted change patch are
   expanded to function-level context, and proxies are computed for
   all functions.
"""
import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from llm_agent_evaluation.data.patch_utils import Chunk, Patch
from llm_agent_evaluation.scorers.lexical import LexicalScorer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lexical Evalutation of Agent Workflows')

    ## Pipeline arguments
    parser.add_argument('--path_to_assets', type=str, default='../resources',
                        help="Path to dataset resources.")
    parser.add_argument('--benchmark', type=str, default='swe-bench',
                        choices=['BugsInPy', 'swe-bench'], help='Benchmark dataset')
    parser.add_argument('--type', type=str, default='Lite',
                        choices=['Lite', 'test'], help='SWE-Bench variant')
    parser.add_argument('--context_key', type=str, default='none',
                        choices=['none', 'function', 'dependency'],
                        help="Amount of context to include in patches")
    parser.add_argument('--patch_label', type=str,
#                        default='20240509_amazon-q-developer-agent-20240430-dev',
                        default='20240617_factory_code_droid',
                        help="Type of patch to score (e.g., 'gold', 'perturbation', etc.)")
    parser.add_argument('--model_name', default='opus-3',
                        choices=['opus-3', 'sonnet-3', 'sonnet-3-5'], help='Claude model key')

    args = parser.parse_args()
    path_to_assets = Path(args.path_to_assets).resolve()

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

    logger.info('Computing semantics-specific proxies for diff patches.')

    metrics = {}

    resolved_status_file = f'{benchmark}.{args.patch_label}.resolved-status.json'
    with open(str(path_to_assets / 'cache' / resolved_status_file), 'r') as f:
        resolved_status_mapper = json.load(f)

    lexical_scorer = LexicalScorer(benchmark=benchmark, to_path=path_to_assets)

    for input_patch in tqdm(patches):
        gold_patch = patch_id_to_gold_patch_mapping[input_patch.id]
        patch_scores = lexical_scorer.score(input_patch, gold_patch)        

        if input_patch.id in resolved_status_mapper['resolved']:
            patch_scores['build_status'] = 'pass'
        else:
            patch_scores['build_status'] = 'fail'

        msg = f'*** Patch {input_patch.id} ***'
        for key, score in patch_scores.items():
            msg += f'\n\t{key}: {score}'
        logger.info(msg)

        metrics[input_patch.id] = deepcopy(patch_scores)

    filename = f'{benchmark}.{args.patch_label}.lexical.json'
    Path(path_to_assets / f'results-{args.model_name}').mkdir(parents=True, exist_ok=True)
    with open(str(path_to_assets / f'results-{args.model_name}' / filename), 'w') as f:
        json.dump(metrics, f, indent=2)
