"""Contains utilities to apply predicted patches from agentic workflows to create
corresponding repository snapshots. Currently supports only SWE-Bench, not BugsInPy.

Log parsing utilities adopted from: https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/harness/log_parsers.py

Usage:
    $ python agentic_patches.py --path_to_assets <path_to_assets>  --agent_name <agent_name> --label <Lite|test>

Example:
    $ python agentic_patches.py --path_to_assets ../resources  --agent_name 20240509_amazon-q-developer-agent-20240430-dev --label Lite
"""
import argparse
import json
import logging
import os
import pathlib
import pickle
import re
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

import git

from tqdm import tqdm

from llm_agent_evaluation.data.patch_utils import (
    Patch,
    apply_patch_to_repo,
    create_change_patch_with_no_context,
    create_change_patch_with_function_context,
)
from llm_agent_evaluation.data.swe_bench.collect import SWEBenchPatchCollector


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Only those agents are recorded that were tested for both SWE-Bench,
# and the SWE-Bench_Lite versions.
ALLOWED_AGENTS = [
    '20231010_rag_claude2',
    '20231010_rag_gpt35',
    '20231010_rag_swellama13b',
    '20231010_rag_swellama7b',
    '20240402_rag_claude3opus',
    '20240402_rag_gpt4',
    '20240402_sweagent_claude3opus',
    '20240402_sweagent_gpt4',
    '20240509_amazon-q-developer-agent-20240430-dev',
    '20240615_appmap-naive_gpt4o',
    '20240617_factory_code_droid',
    '20240702_codestory_aide_mixed',
    '20240620_sweagent_claude3.5sonnet',
    '20240820_honeycomb',
    '20240627_abanteai_mentatbot_gpt4o',
    '20240811_gru'
]


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


def parse_log_pytest(log: str) -> Dict[str, str]:
    """Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> Dict[str, str]:
    """Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if option.startswith("/") and not option.startswith("//") and "*" not in option:
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> Dict[str, str]:
    """Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")

    prev_test = None
    for line in lines:
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = TestStatus.PASSED.value

        # Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        pass_suffixes = (" ... ok", " ... OK", " ...  OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value

        if line.lstrip().startswith("ok") and prev_test is not None:
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$"
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
    return test_status_map


def parse_log_pytest_v2(log: str) -> Dict[str, str]:
    """Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str) -> Dict[str, str]:
    """Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f" {TestStatus.PASSED.value} " in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str) -> Dict[str, str]:
    """Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


def parse_log_matplotlib(log: str) -> Dict[str, str]:
    """Parser for test logs generated with PyTest framework

    Args:
        log (str): log content

    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_pytest_v2,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_pytest,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_pytest,
    "psf/requests": parse_log_pytest_options,
    "pvlib/pvlib-python": parse_log_pytest,
    "pydata/xarray": parse_log_pytest,
    "pydicom/pydicom": parse_log_pytest_options,
    "pylint-dev/astroid": parse_log_pytest,
    "pylint-dev/pylint": parse_log_pytest_options,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pytest,
    "scikit-learn/scikit-learn": parse_log_pytest_v2,
    "sqlfluff/sqlfluff": parse_log_pytest,
    "sphinx-doc/sphinx": parse_log_pytest_v2,
    "sympy/sympy": parse_log_sympy,
}


def get_agentic_patches(
    agent_name: str,
    label: str,
    path_to_temp_dir: pathlib.Path,
) -> List[Dict]:
    """

    Args:
        agent_name: Name of the agent for which the patches should be retrieved.
        label: Label of the SWE-Bench agentic patches (e.g., 'lite', 'test').
        path_to_temp_dir: Path where agentic patches source is temporarily stored.

    Returns:
        List of dictionaries containing 'instance_id', 'model_name_or_path',
        and the 'model_patch'.
    """
    preds_suffix = f'evaluation/{label.lower()}/{agent_name}/all_preds.jsonl'
    path_to_agentic_patches = pathlib.Path(path_to_temp_dir) / preds_suffix

    # Read all agentic patches.
    logger.info('Retrieving agentic patches from source.')
    with open(path_to_agentic_patches, 'r') as json_file:
        all_json_objects = list(json_file)

    return [
        json.loads(json_string) for json_string in all_json_objects
    ]


def parse_logs_and_cache_test_status_maps(
    agent_name: str,
    benchmark: str,
    label: str,
    path_to_cache: pathlib.Path,
    path_to_temp_dir: pathlib.Path,
) -> None:
    """Parse and extract test status maps from test logs for all agentic patches.

    Args:
        agent_name: Name of the agent for which the patches should be retrieved.
        benchmark: SWE-Bench variant.
        label: Label of the SWE-Bench agentic patches (e.g., 'lite', 'test').
        path_to_cache: Path to save test and resolved status maps.
        path_to_temp_dir: Path where agentic patches source is temporarily stored.
    """
    logs_suffix = f'evaluation/{label.lower()}/{agent_name}/logs'
    path_to_logs = path_to_temp_dir / logs_suffix

    logger.info('Parsing all agentic patch logs to extract test status maps.')
    test_status_maps = {}

    # Old log folder formating 
    if any(name.endswith("log") for name in os.listdir(path_to_logs)):
        logger.info("Log folder structure of type 1")

        for name in os.listdir(folder := path_to_logs):
            if name.endswith("log"):
                # SWE-Bench follows a naming convention with '.' as a delimiter:
                # Name: <instance-id>.<agent_name>.eval.log
                instance_id = name.split('.')[0]
                project_name = '-'.join(instance_id.split('-')[:-1]).replace('__', '/')
                with open(os.path.join(folder, name)) as fp:
                    log = fp.read()

                log_parsing_fn = MAP_REPO_TO_PARSER[project_name]
                test_status_maps.setdefault(project_name, {})[instance_id] = log_parsing_fn(log)

    else:
        logger.info("Log folder structure of type 2")
        # New log folder formating
        for instance_id in os.listdir(folder := path_to_logs):
            test_output_path = path_to_logs / f'{instance_id}' / 'test_output.txt'

            if test_output_path.exists():
                project_name = '-'.join(instance_id.split('-')[:-1]).replace('__', '/')
                with open(test_output_path) as fp:
                    log = fp.read()

                log_parsing_fn = MAP_REPO_TO_PARSER[project_name]
                test_status_maps.setdefault(project_name, {})[instance_id] = log_parsing_fn(log)

    # Save test status maps in cache.
    _filename = f'{benchmark}.{agent_name}.test-status.json'
    logger.info(f'Saving test status maps for {agent_name} to {path_to_cache / _filename}')
    with open(str(path_to_cache / _filename), 'w') as f:
        json.dump(test_status_maps, f, indent=2)


def cache_resolved_status_maps(
    agent_name: str,
    benchmark: str,
    label: str,
    path_to_cache: pathlib.Path,
    path_to_temp_dir: pathlib.Path,
) -> None:
    """Extract and cache resolved status maps for all agentic patches.

    Args:
        agent_name: Name of the agent for which the patches should be retrieved.
        benchmark: SWE-Bench variant.
        label: Label of the SWE-Bench agentic patches (e.g., 'lite', 'test').
        path_to_cache: Path to save test and resolved status maps.
        path_to_temp_dir: Path where agentic patches source is temporarily stored.
    """
    results_suffix = f'evaluation/{label.lower()}/{agent_name}/results/results.json'
    shutil.copy2(
        path_to_temp_dir / results_suffix,
        path_to_cache / f'{benchmark}.{agent_name}.resolved-status.json'
    )


def process_agentic_patches_source(
    agent_name: str,
    benchmark: str,
    label: str,
    path_to_cache: pathlib.Path,
) -> List[Dict]:
    """Process agent generation files in SWE_Bench's 'experiments' repository to:
    extract and cache test and resolved status maps; and retrieve agentic patches.

    Args:
        agent_name: Name of the agent for which the patches should be retrieved.
        benchmark: SWE-Bench variant.
        label: Label of the SWE-Bench agentic patches (e.g., 'lite', 'test').
        path_to_cache: Path to save test and resolved status maps.

    Returns:
        List of dictionaries containing 'instance_id', 'model_name_or_path',
        and the 'model_patch'.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_repo = git.Repo.clone_from(
            'https://github.com/swe-bench/experiments.git',
            temp_dir,
        )

        # Retrieve agentic patches.
        agentic_patches = get_agentic_patches(
            agent_name=agent_name,
            label=label,
            path_to_temp_dir=pathlib.Path(temp_dir),
        )

        # Parse and extract test status maps from logs.
        parse_logs_and_cache_test_status_maps(
            agent_name=agent_name,
            benchmark=benchmark,
            label=label,
            path_to_cache=path_to_cache,
            path_to_temp_dir=pathlib.Path(temp_dir),
        )

        # Parse and extract resolved status maps.
        cache_resolved_status_maps(
            agent_name=agent_name,
            benchmark=benchmark,
            label=label,
            path_to_cache=path_to_cache,
            path_to_temp_dir=pathlib.Path(temp_dir),
        )

    return agentic_patches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate agentic patches')

    ## Pipeline arguments
    parser.add_argument('--path_to_assets', type=str, default='../resources',
                        help="Path to dataset resources.")
    parser.add_argument('--agent_name', type=str,
#                        default='20240509_amazon-q-developer-agent-20240430-dev',
                        default='20240617_factory_code_droid',
                        help="Name of agent for which, the patches need to be integrated.")
    parser.add_argument('--label', type=str, default='Lite', choices=['Lite', 'test'],
                        help="SWE-Bench variant (e.g., 'Lite', 'test').")
    args = parser.parse_args()

    benchmark = 'swe-bench_Lite' if args.label == 'Lite' else 'swe-bench'
    path_to_assets = pathlib.Path(args.path_to_assets).resolve()
    path_to_benchmark = path_to_assets / benchmark

    if args.agent_name not in ALLOWED_AGENTS:
        raise ValueError(
            f'Agent {agent_name} is not allowed. Allowed agents are: {ALLOWED_AGENTS}'
        )

    # Get gold patches -- with these, we can avoid building `TestPatch` objects again
    # since they should be consistent across agentic workflows.
    patch_collector = SWEBenchPatchCollector(path_to_assets)
    gold_patches = patch_collector('none', 'gold')
    patch_id_to_gold_patch_mapping = {patch.id: patch for patch in gold_patches}

    agentic_patches = process_agentic_patches_source(
        args.agent_name,
        benchmark,
        args.label,
        path_to_assets / 'cache'
    )

    # Creating `Patch` objects for all agentic patches, with and without context.
    agentic_patches_with_no_context = []
    agentic_patches_with_function_context = []
    for patch_object in tqdm(agentic_patches):
        instance_id = patch_object['instance_id']
        project_name = '-'.join(instance_id.split('-')[:-1]).split('__')[1]
        path_to_instance = path_to_benchmark / 'projects' / project_name / 'instances' / instance_id

        if not path_to_instance.exists():
            logger.info(f"Skipping {instance_id} for {project_name} as it can't be found")
            continue

        with open(str(path_to_instance / 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        diff_patch = patch_object['model_patch']

        # First, create repository snapshots.
        repo_dir = str(path_to_instance / f'snapshots/{args.agent_name}')
        logger.info(f'Creating repository snapshot for {args.agent_name} in {instance_id}')

        if pathlib.Path(repo_dir).is_dir():
            # If repository snapshot already exists, delete it.
            shutil.rmtree(repo_dir)

        agent_repo = git.Repo.clone_from(
            f"https://github.com/{metadata['repo']}.git",
            repo_dir,
        )

        # Apply agent patches only if repository snapshot does not already exist.
        agent_repo.git.reset('--hard', str(metadata['base_commit']))

        logger.info(f'Applying patch to {instance_id}')
        patch_application_result = apply_patch_to_repo(
            patch=diff_patch,
            repo_dir=repo_dir,
            logger=logger
        )
        if not patch_application_result:
            continue

        # Cache `Patch` object created with no additional context.
        agentic_patch_with_no_context = Patch(
            id=instance_id,
            change_patch=create_change_patch_with_no_context(diff_patch),
            test_patch=patch_id_to_gold_patch_mapping[instance_id].test_patch,
            path_to_root=path_to_instance,
        )
        agentic_patches_with_no_context.append(agentic_patch_with_no_context)

        agentic_patches_with_function_context.append(
            Patch(
                id=agentic_patch_with_no_context.id,
                change_patch=create_change_patch_with_function_context(
                    patch=agentic_patch_with_no_context,
                    patch_label=args.agent_name,
                ),
                test_patch=agentic_patch_with_no_context.test_patch,
                path_to_root=agentic_patch_with_no_context.path_to_root,
            )
        )

    # Save `Patch` objects created with no additional context.
    logger.info(f'Saving `Patch` objects with no context for {args.agent_name}')
    _filename = f'{benchmark}.{args.agent_name}.patches.none-context.pkl'
    with open(str(path_to_assets / 'cache' / _filename), 'wb') as f:
        pickle.dump(agentic_patches_with_no_context, f)

    # Save `Patch` objects created with function-level context.
    logger.info(f'Saving `Patch` objects with function-level context for {args.agent_name}')
    _filename = f'{benchmark}.{args.agent_name}.patches.function-context.pkl'
    with open(str(path_to_assets / 'cache' / _filename), 'wb') as f:
        pickle.dump(agentic_patches_with_function_context, f)

    # TODO: If we decide to incorporate dependency-level context, this needs
    # to be updated as well, so as to also cache corresponding patches.
