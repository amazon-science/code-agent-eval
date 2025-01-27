"""Contains utilities to apply perturbed patches and create corresponding
repository snapshots. Can be used for both SWE-Bench and BugsInPy

Usage:
    $ python perturbation.py --path_to_assets <path_to_assets>  --perturb <removal|llm> --benchmark <benchmark-name> --label <Lite|test>

Example:
    $ python perturbation.py --path_to_assets ../resources  --perturb removal --benchmark swe-bench --label Lite

    $ python perturbation.py --path_to_assets ../resources  --perturb removal --benchmark BugsInPy
"""
import argparse
import logging
import pathlib
import pickle
import random
import re
import shutil

from git import Repo

from llm_agent_evaluation.data.patch_utils import (
    Patch,
    apply_patch_to_repo,
    create_change_patch_with_no_context,
    create_change_patch_with_function_context,
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Set seed.
random.seed(42)


def remove_random_hunk(patch_str: str) -> str:
    """Given a diff patch, eliminates a hunk from the patch randomly.

    Args:
        patch_str: Patch string.

    Returns:
        Modified patch string.
    """
    # Regex patterns to capture different components of the diff
    file_pattern = r"diff --git a/(.+) b/\1"
    index_pattern = r"index (\w+)\.\.(\w+) (\d+)"
    header_pattern = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@\s+(.*?)(?=(^@@ -)|\Z)"

    # Split the diff text into file sections
    file_sections = re.split(file_pattern, patch_str)[1:]

    if not file_sections: return patch_str

    # file_sections would contain file names, followed by corresponding file
    # sections. Here, we randomly select one of the file sections.
    random_file_content = file_sections[random.choice(range(0, len(file_sections), 2)) + 1]
    # Extract all hunks.
    changes = re.findall(header_pattern, random_file_content, re.DOTALL | re.MULTILINE)
    # Randomly select a hunk for elimination.
    change = random.choice(changes)
    text_to_remove = f"@@ -{change[0]},{change[1]} +{change[2]},{change[3]} @@ {change[4]}"
    modified_patch = patch_str.replace(text_to_remove, '')
    return modified_patch


def perturb_by_removal(original_patch: Patch, path_to_root: pathlib.Path) -> Patch:
    """Create new patch by removing a random hunk from original patch.

    Args:
        original_patch: Original patch to perturb.
        path_to_root: Path to the root of the code repository containing
            all repository snapshots.

    Returns:
        Perturbed patch.
    """
    return Patch(
        id=original_patch.id,
        change_patch=create_change_patch_with_no_context(
            remove_random_hunk(
                original_patch.change_patch.text
            )
        ),
        test_patch=original_patch.test_patch,
        path_to_root=path_to_root,
    )
    

def perturb_with_llm(original_patch: Patch) -> Patch:
    """Create new patch by perturbing original patch with LLM.

    Args:
        original_patch: Original patch to perturb.

    Returns:
        Perturbed patch.
    """
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate perturbed patches')

    parser.add_argument('--path_to_assets', type=str, default='../resources',
                        help="Path to dataset resources.")
    parser.add_argument('--perturb', type=str, default='removal',
                        choices=['removal', 'llm'], help="Perturbation strategy.")
    parser.add_argument('--benchmark', type=str, default='BugsInPy',
                        choices=['BugsInPy', 'swe-bench'], help='Benchmark dataset')
    parser.add_argument('--label', type=str, default='Lite', choices=['Lite', 'test'],
                        help="SWE-Bench variant (e.g., 'Lite', 'test').")
    args = parser.parse_args()

    if args.benchmark == 'BugsInPy':
        benchmark = args.benchmark
    elif args.benchmark == 'swe-bench':
        benchmark = 'swe-bench_Lite' if args.label == 'Lite' else 'swe-bench'

    path_to_assets = pathlib.Path(args.path_to_assets).resolve()
    patches_filename = f'{benchmark}.gold.patches.none-context.pkl'

    with open(str(path_to_assets / 'cache' / patches_filename), 'rb') as f:
        original_patches = pickle.load(f)

    perturb_fn = perturb_by_removal if args.perturb == 'removal' else perturb_with_llm

    perturbed_patches_with_no_context = []
    perturbed_patches_with_function_context = []
    for patch in original_patches:
        project = '-'.join(patch.id.split('-')[:-1])
        path_to_instance = path_to_assets / benchmark / 'projects' / project /'instances' / patch.id
        base_dir = str(path_to_instance / f'snapshots/base')
        perturbation_dir = str(path_to_instance / f'snapshots/perturb-{args.perturb}')

        # Create a copy of the base version to apply patch over.
        logger.info(f'Creating repository snapshot for {patch.id}')
        try:
            shutil.copytree(base_dir, perturbation_dir)
        except FileExistsError:
            shutil.rmtree(perturbation_dir)
            shutil.copytree(base_dir, perturbation_dir)
        # except FileNotFoundError:
        #     print('Skipping')
        #     continue

        # Initialize copy as a GitHub repository.
        perturbation_repo = Repo(perturbation_dir)
        perturbed_patch = perturb_fn(patch, path_to_instance) # Perturb patch...
        perturbed_patches_with_no_context.append(perturbed_patch)

        # Create snapshot for perturbed patches.
        new_patch_str = perturbed_patch.change_patch.text + \
                        '\n' + \
                        perturbed_patch.test_patch.text
        apply_patch_to_repo(patch=new_patch_str, repo_dir=perturbation_dir, logger=logger)

        # Expand to function-level context.
        perturbed_patches_with_function_context.append(
            Patch(
                id=perturbed_patch.id,
                change_patch=create_change_patch_with_function_context(
                    patch=perturbed_patch,
                    patch_label=f'perturb-{args.perturb}'
                ),
                test_patch=perturbed_patch.test_patch,
                path_to_root=perturbed_patch.path_to_root,
            )
        )

    # Save perturbed `Patch` objects created with no additional context.
    logger.info((f'Saving perturbed (by {args.perturb}) `Patch` objects with '
                  'no context.'))
    _filename = f'{benchmark}.perturb-{args.perturb}.patches.none-context.pkl'
    with open(str(path_to_assets / 'cache' / _filename), 'wb') as f:
        pickle.dump(perturbed_patches_with_no_context, f)

    # Save perturbed `Patch` objects created with function-level context.
    logger.info((f'Saving perturbed (by {args.perturb}) `Patch` objects with '
                  'function-level context.'))
    _filename = f'{benchmark}.perturb-{args.perturb}.patches.function-context.pkl'
    with open(str(path_to_assets / 'cache' / _filename), 'wb') as f:
        pickle.dump(perturbed_patches_with_function_context, f)
