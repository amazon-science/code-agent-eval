import json
import logging
import pathlib
import shutil
import tempfile

from datasets import load_dataset

from git import Repo

from tqdm import tqdm

from llm_agent_evaluation.data import BaseDataBuilder
from llm_agent_evaluation.data.patch_utils import apply_patch_to_repo


class SWEBenchDataBuilder(BaseDataBuilder):
    """A data builder class for the SWE-Bench dataset.

    This class is responsible for building the SWE-Bench dataset from scratch,
    which includes cloning the repository and creating snapshots of both pre-commit
    (i.e., 'base') and post-commit (i.e., 'gold') versions of the repository for
    all task instances in each project.
    """
    def __init__(self, name: str, suffix: str):
        super().__init__()
        self.name = name

    def __call__(self, to_path: pathlib.Path):
        self.init_build_logger(to_path)
        self.build(to_path)

    def init_build_logger(self, to_path: pathlib.Path):
        """Initialize a logger that saves all SWE-Bench-specific data building
        logs to its corresponding 'build_logs.txt' file.

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

    def build(self, to_path: pathlib.Path):
        """Build SWE-Bench dataset from scratch. Includes cloning the repo, and
        creating repo snapshots for all task instances in each project.

        Args:
            to_path: Path-like object to where the repository should be built in.
        """
        if not hasattr(self, 'logger'):
            self.init_build_logger(to_path)

        # Store all project instances in 'projects/<project-name>/instances/<instance-id>
        (to_path / 'projects').mkdir(exist_ok=True, parents=True)

        data_instances = load_dataset(f'princeton-nlp/{self.name}', split='test')

        for instance in tqdm(data_instances):
            project_name = str(instance['repo']).split('/')[1]
            instance_id = str(instance['instance_id'])

            self.logger.info(f'Checking out {instance_id} in {project_name} project.')

            path_to_instance = to_path / 'projects' / project_name / 'instances' / instance_id

            # Check if instance has already been set up, and if so, skip.
            if path_to_instance.is_dir():
                if (path_to_instance / 'snapshots' / 'base').is_dir() and \
                   (path_to_instance / 'snapshots' / 'gold').is_dir():
                    continue
                else:
                    shutil.rmtree(str(path_to_instance))

            try:
                path_to_instance.mkdir(exist_ok=True, parents=True)
                metadata = {
                    k: v for k, v in instance.items() if str(k) not in ['patch', 'test_patch']
                }

                with open(path_to_instance / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Across benchmarks, change and test patches are saved together in
                # <project-name>/instances/<instance-id>/gold_patch.txt
                # Such a project-level stratification helps generalize utilities across benchmarks.
                change_patch, test_patch = instance['patch'], instance['test_patch']
                combined_patch = change_patch + '\n' + test_patch
                with open(path_to_instance / 'gold_patch.txt', 'w') as f:
                    f.write(combined_patch)

                with open(path_to_instance / 'gold_change_patch.txt', 'w') as f:
                    f.write(change_patch)

                with open(path_to_instance / 'gold_test_patch.txt', 'w') as f:
                    f.write(test_patch)

                # Create/save 'base' and 'gold' snapshots.
                (path_to_instance / 'snapshots').mkdir(exist_ok=True, parents=True)

                repo_url = f"https://github.com/{instance['repo']}.git"
                base_repo = Repo.clone_from(
                    repo_url,
                    str(path_to_instance / 'snapshots/base'),
                )
                base_repo.git.reset('--hard', str(instance['base_commit']))

                gold_repo = Repo.clone_from(
                    repo_url,
                    str(path_to_instance / 'snapshots/gold'),
                )
                gold_repo.git.reset('--hard', str(instance['base_commit']))

                modified_files, _ = apply_patch_to_repo(
                    combined_patch,
                    str(path_to_instance / 'snapshots/gold'),
                    self.logger,
                )
            except:
                self.logger.exception(f'Error building {instance_id}. Skipping...')
                shutil.rmtree(str(path_to_instance))
                continue
