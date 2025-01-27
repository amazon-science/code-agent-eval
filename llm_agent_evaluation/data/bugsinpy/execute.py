import enum
import logging
import os
import pathlib
import shutil
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

from llm_agent_evaluation.utils import CustomEnumMeta


class TestRunner:
    """A test runner class for running all relevant (e.g., bug-triggering)
    tests in BugsInPy benchmark.
    """
    name = 'BugsInPy'

    def __init__(self, path_to_datasets: pathlib.Path):
        """Initialize `TestRunner` class.

        Args:
            path_to_datasets: Path to all dataset resources.
        """
        self.path_to_benchmark = path_to_datasets / self.name

    def init_test_logger(
        self,
        path_to_datasets: pathlib.Path,
        project: str,
        patch_label: str,
    ) -> None:
        """Initialize a logger that saves all BugsInPy benchmark-specific
        unit test logs to its corresponding 'test_logs.txt' file.

        Args:
            path_to_datasets: Path to all dataset resources.
            project: Project in BugsInPy benchmark.
            patch_label: Identifier for the type of patch. For example, 'gold'
                indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.
        """
        path_to_logs = path_to_datasets / 'logs' / self.name
        path_to_logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(path_to_logs / f'tests_{patch_label}_{project}.txt'),
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def _test_single_bug(
        self,
        project: str,
        bug_id: int,
        patch_label: str = 'gold',
    ) -> Dict[str, bool]:
        """Test single bug in a BugsInPy project.

        Args:
            project: Project in BugsInPy benchmark.
            bug_id: Bug identifier.
            patch_label: Identifier for the type of patch. For example, 'gold'
                indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.

        Returns:
            Test results for single bu in BugsInPy project.
        """
        path_to_bug = self.path_to_benchmark / 'projects' / project / 'bugs' / str(bug_id)

        # Copy requirements files
        for file in ['bug.info', 'requirements.txt', 'setup.sh', 'run_test.sh']:
            shutil.copy(
                str(path_to_bug / file),
                str(path_to_bug / f'snapshots/{patch_label}/bugsinpy_{file}'),
            )

        subprocess.run(['bash', '-e', 'run_sandbox.sh', str(path_to_bug), patch_label])

        for file in ['requirements.txt', 'setup.sh', 'run_test.sh']:
            os.remove(str(path_to_bug / f'snapshots/{patch_label}/bugsinpy_{file}'))

    def _test_all_bugs(
        self,
        project: str,
        patch_label: str = 'gold',
    ) -> Dict[int, Dict[str, bool]]:
        """Test all bugs in a BugsInPy project.

        Args:
            project: Project in BugsInPy benchmark.
            patch_label: Identifier for the type of patch. For example, 'gold'
                indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.

        Returns:
            Test results for all bugs in a BugsInPy project.
        """
        self.init_test_logger(self.path_to_benchmark.parent, project, patch_label)
        self.logger.info(f'Running unit tests for all bugs in {project} project.')
        all_bugs = list((self.path_to_benchmark / 'projects' / project / 'bugs').iterdir())

        results = {}
        for item in all_bugs_in_project:
            if not item.is_dir():
                continue

            bug_id = item.name
            self.logger.info(f'\t*** Bug {bug_id} in {project} ***')
            results[bug_id] = self._test_single_bug(project, bug_id, patch_label)
        return results

    def test(
        self,
        project: str,
        bug_id: int = None,
        patch_label: str = 'gold',
    ) -> Union[Dict[str, bool] or Dict[int, Dict[str, bool]]]:
        """Setup and run test(s) for BugsInPy benchmark in a sandbox environment.

        Args:
            project: Project in BugsInPy benchmark.
            bug_id: Bug identifier.
            patch_label: Identifier for the type of patch. For example, 'gold'
                indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.

        Returns:
            Test results for one or all bugs in a BugsInPy project.
        """
        if not bug_id:
            self._test_all_bugs(project, patch_label)
        else:
            self._test_single_bug(project, bug_id, patch_label)


# if __name__ == '__main__':
#     TODO: Add your local path bellow 
#     runner = TestRunner(pathlib.Path('/my/local/path/llm_agent_evaluation/resources'))

#     PROJECTS = [
#         'PySnooper',
#         'ansible',
#         'black'
#         'cookiecutter',
#         'fastapi',
#         'httpie',
#         'keras',
#         'luigi',
#         'matplotlib',
#         'pandas',
#         'sanic',
#         'scrapy',
#         'spacy',
#         'thefuck',
#         'tornado',
#         'tqdm',
#         'youtube-dl'
#     ]

#     for project in PROJECTS:
#         results = runner.test(project, 1, 'gold')
