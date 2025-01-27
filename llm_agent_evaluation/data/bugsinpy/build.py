import logging
import pathlib
import shutil
import tempfile
from collections import namedtuple
from copy import deepcopy
from typing import List, NamedTuple

from git import Repo

from tqdm import tqdm

from llm_agent_evaluation.data import BaseDataBuilder
from llm_agent_evaluation.data.bugsinpy.error import (
    InvalidProjectError,
    IncompleteProjectInfoError,
    IncompleteBugInfoError,
)


class ProjectInfo(NamedTuple):
    github_url: str
    status: str


class BugInfo(NamedTuple):
    python_version: str
    buggy_commit_id: str
    fixed_commit_id: str
    test_files: List[str]


class BugsInPyDataBuilder(BaseDataBuilder):
    """A data builder class for the BugsInPy dataset.

    This class is responsible for building the BugsInPy dataset from scratch,
    which includes cloning the repository and creating snapshots of both buggy
    and fixed versions of the repository for all bugs in each project.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __call__(self, repo_url: str, to_path: pathlib.Path, projects: List[str]):
        self.init_build_logger(to_path)
        self.build(repo_url, to_path, projects)

    def _extract_project_info(self, path_to_project: pathlib.Path) -> ProjectInfo:
        """Extracts project information necessary for setup.

        Args:
            path_to_project: Path to a project in BugsInPy benchmark.

        Returns:
            ProjectInfo: A named tuple containing the GitHub URL for the project,
            and its status, i.e., whether it is OK to use or not.
        """
        # Check project status.
        with open(str(path_to_project / 'project.info'), 'r') as f:
            project_info = f.readlines()

        for line in project_info:
            key, value = line.split('=')
            if key == 'github_url':
                github_url = deepcopy(value.strip()[1:-1])
            elif key =='status':
                status = deepcopy(value.strip()[1:-1])

        if github_url and status:
            return ProjectInfo(github_url=github_url, status=status)
        else:
            raise IncompleteProjectInfoError(path_to_project, self.logger)

    def _extract_bug_info(self, path_to_bug: pathlib.Path) -> BugInfo:
        """Extracts all bug-specific information.

        Args:
            path_to_bug: Path to a bug in BugsInPy benchmark.

        Returns:
            BugInfo: A named tuple containing all bug-specific information,
            including the Python version, buggy commit ID, fixed commit ID,
            and the test files to reproduce the bug.
        """
        with open(str(path_to_bug / 'bug.info'), 'r') as f:
            bug_info = f.readlines()

        for line in bug_info:
            key, value = line.split('=')
            if key.strip() == 'python_version':
                python_version = deepcopy(value.strip()[1:-1])
            elif key.strip() == 'buggy_commit_id':
                buggy_commit_id = deepcopy(value.strip()[1:-1])
            elif key.strip() == 'fixed_commit_id':
                fixed_commit_id = deepcopy(value.strip()[1:-1])
            elif key.strip() == 'test_file':
                test_files = deepcopy(value.strip()[1:-1])
                test_files = test_files.split(';')
                # In some bug.info files, the 'test_file' value has a trailing ';',
                # which can cause test mismatch issues later.
                # We address this by validating the extracted test file paths.
                test_files = [file for file in test_files if 'test' in str(file)]

        if python_version and buggy_commit_id and fixed_commit_id and test_files:
            return BugInfo(
                python_version=python_version,
                buggy_commit_id=buggy_commit_id,
                fixed_commit_id=fixed_commit_id,
                test_files=test_files
            )
        else:
            raise IncompleteBugInfoError(path_to_bug, self.logger)

    def init_build_logger(self, to_path: pathlib.Path):
        """Initialize a logger that saves all BugsInPy benchmark-specific data
        building logs to its corresponding 'build_logs.txt' file.

        Args:
            to_path: Path-like object to where all benchmark resources are saved.
        """
        path_to_build_logs = to_path.parent / 'logs' / self.name
        path_to_build_logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(path_to_build_logs / 'build_logs.txt'),
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def build(self, repo_url: str, to_path: pathlib.Path, projects: List[str]):
        """Build BugsInPy dataset from scratch. Includes cloning the repo, and
        creating repo snapshots for all bugs in each project.

        Args:
            repo_url: Valid git url.
            to_path: Path-like object to where the repository should be built in.
            projects: List of all project names in BugsInPy benchmark.
        """
        if not hasattr(self, 'logger'):
            self.init_build_logger(to_path)

        # Download contents from repository URL.
        repo = Repo.clone_from(repo_url, str(to_path))

        # Extract all bug IDs.
        all_projects = list((to_path / 'projects').iterdir())
        for project in tqdm(all_projects):
            try:
                if not project.stem in projects:
                    continue

                # Rename 'bugs' to 'instances' to ensure consistency across benchmarks.
                old_dir = project / 'bugs'
                new_dir = project / 'instances'
                old_dir.rename(new_dir)

                github_url, status = self._extract_project_info(project)

                # Skip instance if project status is not okay.
                if status != 'OK':
                    raise InvalidProjectError(project.stem, self.logger)

                # Cloning repo for each bug using the GitHub URL can be time-consuming.
                # Instead, we can place the clone in a temporary directory, and use
                # this to create repository snapshots for all bugs.
                self.logger.info(f'Cloning {project.stem} project into temporary directory.')
                with tempfile.TemporaryDirectory() as temp_dir:
                    project_repo = Repo.clone_from(github_url, temp_dir)

                    all_bugs_in_project = list((project / 'instances').iterdir())
                    for item in tqdm(all_bugs_in_project):
                        try:
                            if not item.is_dir():
                                continue

                            bug_id = item.name
                            self.logger.info(f'Checking out bug {bug_id} in {project.stem} project.')

                            # Rename 'bug_patch.txt' to 'patch.txt'.
                            old_file = project / 'instances' / bug_id / 'bug_patch.txt'
                            new_file = project / 'instances' / bug_id / 'gold_patch.txt'
                            old_file.rename(new_file)

                            # Extract bug info.
                            bug_info = self._extract_bug_info(item)
                            path_to_snapshots = project / 'instances' / bug_id / 'snapshots'
                            path_to_snapshots.mkdir(exist_ok=True)

                            # Create buggy repo snapshot by resetting HEAD to corresponding commit IDs.
                            shutil.copytree(temp_dir, str(path_to_snapshots / 'base'))
                            buggy_repo = Repo(path_to_snapshots / 'base')
                            buggy_repo.git.reset('--hard', bug_info.buggy_commit_id)

                            # Create fixed repo snapshot by resetting HEAD to corresponding commit IDs.
                            shutil.copytree(temp_dir, str(path_to_snapshots / 'gold'))
                            fixed_repo = Repo(path_to_snapshots / 'gold')
                            fixed_repo.git.reset('--hard', bug_info.fixed_commit_id)

                            # Copy changed test files from fixed version to buggy version.
                            for test_file in bug_info.test_files:
                                shutil.copy2(
                                    str(path_to_snapshots / 'gold' / test_file),
                                    str(path_to_snapshots / 'base' / test_file)
                                )

                            # Rename 'bug_id' with a more generic 'instance_id
                            instance_id = f'{project.name}-{str(bug_id).zfill(5)}'
                            path_to_instance = item.parent / instance_id
                            item.rename(path_to_instance)

                        except IncompleteBugInfoError:
                            # Skipping invalid bugs after logging the Bug ID.
                            continue

            except (IncompleteProjectInfoError, InvalidProjectError):
                # Skipping invalid projects after logging the project name.
                continue
