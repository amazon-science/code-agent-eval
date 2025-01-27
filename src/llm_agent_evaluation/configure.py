"""Configures and sets up BugsInPy and SWE-Bench benchmarks, creating patches
with no additional context, and function-level context.

1. For BugsInPy benchmark:
    $ python configure.py --benchmark BugsInPy

2. For SWE-Bench:
    $ python configure.py --benchmark swe-bench --type {Lite|test}
"""
import argparse
import logging
from pathlib import Path

from llm_agent_evaluation.data.bugsinpy import BugsInPyPatchCollector
from llm_agent_evaluation.data.swe_bench import SWEBenchPatchCollector


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare resources for LLM-Based Evalutation of Agentic Workflows'
    )

    ## Pipeline arguments
    parser.add_argument('--path_to_assets', type=str, default='resources',
                        help="Path to dataset resources.")
    parser.add_argument('--benchmark', type=str, default='BugsInPy',
                        choices=['BugsInPy', 'swe-bench'], help='Benchmark dataset')
    parser.add_argument('--type', type=str, default='Lite',
                        choices=['Lite', 'test'], help='SWE-Bench variant')

    args = parser.parse_args()

    path_to_assets = Path(args.path_to_assets).resolve()

    if args.benchmark == 'BugsInPy':
        patch_collector = BugsInPyPatchCollector
        _args = [path_to_assets]
    elif args.benchmark == 'swe-bench':
        patch_collector = SWEBenchPatchCollector
        _args = [path_to_assets, args.type]

    # Collecting the benchmarks also builds and sets up the benchmark datasets.
    # TODO: Include 'dependency' as a context_key here if we choose to extend
    # and add dependence-level context.
    for context_key in ['none', 'function']:
        _ = patch_collector(*_args)(context_key, 'gold')
