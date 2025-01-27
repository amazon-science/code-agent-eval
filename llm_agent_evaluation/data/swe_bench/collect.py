import pathlib
from typing import List

from llm_agent_evaluation.data import BaseDataCollector
from llm_agent_evaluation.data.patch_utils import (
    Patch,
    Context,
    TestPatch,
    parse_git_diff_enhanced,
)
from llm_agent_evaluation.data.constants import SWEBENCH_PROJECTS
from llm_agent_evaluation.data.swe_bench.build import SWEBenchDataBuilder
from llm_agent_evaluation.external.ast.explorer import (
    build_ast_from_source,
    find_nodes_of_type,
)


def create_test_patch_for_swebench(path_to_instance: pathlib.Path) -> TestPatch:
    """Creates a ``TestPatch`` object from recorded test patch.

    Args:
        path_to_instance: Path to an instance in SWE-Bench.

    Returns:
        ``TestPatch`` object containing all test chunks.
    """
    with open(str(path_to_instance / 'gold_test_patch.txt'), 'r') as f:
        test_patch = f.read()

    path_to_repo = path_to_instance / 'snapshots/gold'

    relevant_tests = []
    for diff in parse_git_diff_enhanced(test_patch):
        file_name = diff['filename']

        with open(str(path_to_repo / file_name), 'r') as f:
            source = f.read()

        source_lines = source.split('\n')
        tree = build_ast_from_source(source)
        class_def_nodes = list(find_nodes_of_type(tree, 'class_definition'))
        func_def_nodes = list(find_nodes_of_type(tree, 'function_definition'))

        unique_headers = set()
        relevant_tests_in_file = []
        # The strategy here is to extract the name of the unit tests in the test
        # patch by comparing the corresponding diff header with all function names
        # in the containing file, and to extract corresponding method body.
        for change in diff['changes']:
            header = change['text'].split('\n')[0]
            if header in unique_headers: continue
            else: unique_headers.add(header)

            containing_node = None
            for node in class_def_nodes + func_def_nodes:
                definition_text = source_lines[node.start_point[0]].strip()
                if header.strip() in definition_text:
                    containing_node = node
                    break

            # Consider hunks only in function/class-level context.
            if not containing_node:
                continue

            start_lineno = containing_node.start_point[0]
            end_lineno = containing_node.end_point[0]
            
            relevant_tests_in_file.append(
                '\n'.join(source_lines[start_lineno: end_lineno+1])
            )

        # If say a new test file was added, it's diff header would be empty
        # and the above strategy would not work. In this case, we simply
        # extract all tests from the file.
        if not relevant_tests_in_file:
            relevant_tests_in_file = [
                '\n'.join(source_lines[node.start_point[0]: node.end_point[0]+1])
                for node in func_def_nodes
            ]
        relevant_tests += relevant_tests_in_file
    return TestPatch(text=test_patch, relevant_tests=relevant_tests)


class SWEBenchPatchCollector(BaseDataCollector):
    """Collect diff patches from SWE-Bench data repository. If not already
    available, it calls the corresponding data builder class and sets it up.
    """
    name = 'swe-bench'

    def __init__(self, path_to_datasets: pathlib.Path, suffix: str = 'Lite'):
        """Initialize patch collection from SWE-Bench data repository.

        Args:
            path_to_datasets: Path to all datasets.
            suffix: Indicates whether to use full 'swe-bench' or 'swe-bench_Lite'.
        """
        super().__init__()
        self.name = f'{self.name}_Lite' if suffix == 'Lite' else self.name
        self.path_to_benchmark = path_to_datasets / self.name

        # Build benchmark from scratch if not already set up.
        if not self.is_available(self.path_to_benchmark):
            builder = SWEBenchDataBuilder(self.name, suffix)
            builder(self.path_to_benchmark)

        self.path_to_cache = path_to_datasets / 'cache'

    def collect(self, context_key: str, patch_label: str) -> List[Patch]:
        """Collect diff patches from SWE-Bench data repository.

        Args:
            context_key: Identifier for the amount of context to include for
                all chunks in a diff patch.
            patch_label: Identifier for the type of patch to collect. For example,
                'gold' indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.

        Returns:
            patches: List of all diff patches in BugsInPy benchmark
                (created as ``Patch`` objects).
        """
        return self._collect(
            context_key=context_key,
            patch_label=patch_label,
            all_project_names=SWEBENCH_PROJECTS,
            test_patch_fn=create_test_patch_for_swebench,
        )
