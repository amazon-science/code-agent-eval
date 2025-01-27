import abc
import pathlib
import pickle
from copy import deepcopy
from typing import Any, List, Tuple

from llm_agent_evaluation.data import BaseDataCollector
from llm_agent_evaluation.data.bugsinpy.build import BugsInPyDataBuilder
from llm_agent_evaluation.data.constants import (
    BUGSINPY_REPO_URL,
    BUGSINPY_PROJECTS,
)
from llm_agent_evaluation.data.patch_utils import (
    Patch,
    Context,
    TestPatch,
    extract_change_and_test_patches,
)
from llm_agent_evaluation.external.ast.explorer import (
    build_ast_from_source,
    find_nodes_of_type,
)


def extract_test_code_from_repo_snapshot(
    path_to_file: pathlib.Path,
    func_name: str,
    class_name: str,
) -> str:
    """Extract test function code from repo snapshot. Note that the failing
    test command can refer to the test file, a class and function within it,
    or a stand-alone function. In each of these cases, the corresponding code
    strings within their scope are returned.

    Args:
        path_to_file: Path to file containing test function code.
        func_name: Name of test function.
        class_name: Name of class containing test function.

    Returns:
        Test code.
    """
    try:
        with open(str(path_to_file), 'r') as f:
            source = f.read()
    except FileNotFoundError:
        raise ValueError(f'Could not find file {path_to_file}')

    source_lines = source.split('\n')

    tree = build_ast_from_source(source)
    class_nodes = list(find_nodes_of_type(tree, 'class_definition'))
    def_nodes = list(find_nodes_of_type(tree, 'function_definition'))

    test_node = None
    if class_name: # If class name is provided...
        for class_node in class_nodes:
            class_def_text = source_lines[class_node.start_point[0]].strip()
            if class_name.strip() in class_def_text:
                class_start_lineno = class_node.start_point[0]
                class_end_lineno = class_node.end_point[0]
                # If class name is provided but the function name is not, extract
                # entire source code in the scope of the test class.                
                if not func_name:
                    test_node = class_node
                    break
                # If both class and function names are provided, extract the
                # source code in the scope of the test function.
                else:
                    for func_node in def_nodes:
                        func_start_lineno = func_node.start_point[0]
                        func_end_lineno = func_node.end_point[0]
                        if class_start_lineno <= func_start_lineno < func_end_lineno <= class_end_lineno:
                            func_def_text = source_lines[func_node.start_point[0]].strip()
                            if func_name.strip() in func_def_text:
                                test_node = func_node
                                break
    else:
        # If class name is not provided but function name is, extract source
        # code in the scope of that test function directly.
        if func_name:
            for func_node in def_nodes:
                func_def_text = source_lines[func_node.start_point[0]].strip()
                if func_name.strip() in func_def_text:
                    test_node = func_node
                    break
    if test_node:
        start_lineno = test_node.start_point[0]
        end_lineno = test_node.end_point[0]
        return '\n'.join(source_lines[start_lineno: end_lineno+1])
    else:
        # If both class and function names are not provided, extract entire
        # source code in the scope of the test file. 
        return source


def create_test_patch_for_bugsinpy(path_to_instance: pathlib.Path) -> TestPatch:
    """Creates a ``TestPatch`` object from the known buggy-tests.

    Args:
        path_to_instance: Path to a bug in BugsInPy benchmark.

    Returns:
        ``TestPatch`` object containing all test chunks.
    """
    path_to_project = path_to_instance.parent.parent

    with open(str(path_to_instance / 'gold_patch.txt'), 'r') as f:
        patch = f.read()
    _, test_patch = extract_change_and_test_patches(patch)

    with open(str(path_to_instance / 'run_test.sh'), 'r') as f:
        test_lines = f.readlines()

    relevant_tests = []
    for line in test_lines:
        # Four types of test utilities are used in BugsInPy benchmark.
        # Skip all lines in `run_test.sh` if it does not correspond to
        # a test command.
        if not (
            'pytest' in line or
            'py.test' in line or
            'unittest' in line or
            'tox' in line
        ):
            continue
        # `unittest` commands are of the form `tests.test_something.test_foo`
        # For all other testing utilities, they are of the form
        # `tests/test_something.py` or `tests/test_something::test_foo`
        split_by = '.' if 'unittest' in line else '::'
        cmd_contents = line.strip().split(' ')[-1].split(split_by)
        path_to_file = pathlib.Path(cmd_contents[0])

        func_name, class_name = None, None
        if len(cmd_contents) == 2:
            # Example: `pytest tests/test_something.py::test_foo`
            func_name = cmd_contents[1]
        elif len(cmd_contents) == 3:
            # Example: `pytest test/test_something.py::Foo::test_bar`
            class_name = cmd_contents[1]
            func_name = cmd_contents[2]

        test_code = extract_test_code_from_repo_snapshot(
            path_to_instance / f'snapshots/gold' / path_to_file,
            func_name,
            class_name,
        )
        relevant_tests.append(test_code)
    return TestPatch(text=test_patch, relevant_tests=relevant_tests)


class BugsInPyPatchCollector(BaseDataCollector):
    """Collect diff patches from BugsInPy benchmark repository. If not
    already available, it calls the corresponding data builder class and
    sets up the BugsInPy benchmark.
    """
    name = 'BugsInPy'

    def __init__(self, path_to_datasets: pathlib.Path):
        """Initialize patch collection from BugsInPy benchmark repository.

        Args:
            path_to_datasets: Path to all datasets.
        """
        super().__init__()
        self.path_to_benchmark = path_to_datasets / self.name

        # Build benchmark from scratch if not already set up.
        if not self.is_available(self.path_to_benchmark):
            builder = BugsInPyDataBuilder(self.name)
            builder(BUGSINPY_REPO_URL, self.path_to_benchmark, BUGSINPY_PROJECTS)

        self.path_to_cache = path_to_datasets / 'cache'

    def collect(self, context_key: str, patch_label: str) -> List[Patch]:
        """Collect diff patches from BugsInPy benchmark repository.

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
            all_project_names=BUGSINPY_PROJECTS,
            test_patch_fn=create_test_patch_for_bugsinpy,
        )
