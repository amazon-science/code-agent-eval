import abc
import pathlib
import pickle
from typing import Any, Callable, List

from llm_agent_evaluation.data.patch_utils import (
    TestPatch,
    Patch,
    Context,
    correct_modified_lines_offset,
    extract_change_and_test_patches,
    merge_chunks_in_header,
    parse_git_diff_enhanced,
    create_change_patch_with_no_context,
    create_change_patch_with_function_context,
    create_change_patch_with_dependency_context,
)


class BaseDataBuilder(abc.ABC):
    """Abstract base class for setting up repository-level benchmark datasets
    (e.g., BugsInPy, SWE-Bench, etc.) from source URLs.
    """

    def __call__(self) -> None:
        self.build() 

    @abc.abstractmethod
    def build(self) -> None:
        """Build datasets from scratch for agentic workflows."""
        pass


class BaseDataCollector(abc.ABC):
    """Abstract base class for processing benchmarks to collect agentic
    workflow-specific data (e.g., patches, repository snapshots, etc.).

    These utilities are designed to be standard across benchmark datasets.
    """
    def __call__(self, context_key: str, patch_label: str) -> List[Patch]:
        return self.collect(context_key, patch_label)

    def _load_patches(self, context_key: str, patch_label: str) -> List[Patch]:
        """Loads patches from cache.

        Args:
            context_key: Identifier for the amount of context to include for
                all chunks in a diff patch.
            patch_label: Identifier for the type of patch to collect. For example,
                'gold' indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.

        Returns:
            patches: List of all patches in benchmark (created as ``Patch`` objects).
        """
        filename = f'{self.name}.{patch_label}.patches.{context_key}-context.pkl'
        with open(str(self.path_to_cache / filename), 'rb') as f:
            patches = pickle.load(f)
        return patches

    def _save_patches(
        self,
        patches: List[Patch],
        context_key: str,
        patch_label: str
    ) -> None:
        """Save patches to cache.

        Args:
            patches: List of patches.
            context_key: Identifier for the amount of context to include for
                all chunks in a diff patch.
            patch_label: Identifier for the type of patch to collect. For example,
                'gold' indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.
        """
        filename = f'{self.name}.{patch_label}.patches.{context_key}-context.pkl'
        with open(str(self.path_to_cache / filename), 'wb') as f:
            pickle.dump(patches, f)

    def _create_patch_with_no_context(
        self,
        change_text: str,
        path_to_root: pathlib.Path,
        test_patch_fn: Callable,
        ) -> Patch:
        """Create ``Patch`` object for an input diff patch string with no context,
        i.e., where each (before/after) chunk corresponds to a hunk.

        Args:
            change_text: Diff patch string, including both change and test patches.
            path_to_root: Path to the root of the repository (e.g., bug in BugsInPy,
                or task instance in SWE-Bench).
            test_patch_fn: Callable to create test patches. This needs to be defined
                independently for each `PatchCollector` object.

        Returns:
            ``Patch`` object containing all chunks with no context.
        """
        instance_id = path_to_root.name
        project = path_to_root.parent.parent.name

        change_patch_string, test_patch_string = extract_change_and_test_patches(change_text)
        return Patch(
            id=path_to_root.name,
            change_patch=create_change_patch_with_no_context(
                change_patch_string,
            ),
            test_patch=test_patch_fn(path_to_root),
            path_to_root=path_to_root,
        )

    def _create_patch_with_context(
        self,
        patch_with_no_context: Patch,
        context_key: str
    ) -> Patch:
        """Create ``Patch`` object for an input diff patch string with function
        context, i.e., expanded until function/class-level, or with dependency
        context, i.e., also including 'calls' and 'called_by' relations.

        Args:
            change_text: Diff patch string, including both change and test patches.
            context_key: Identifier for the amount of context to include for
                all chunks in a diff patch.

        Returns:
            ``Patch`` object containing all chunks with no context.
        """
        if context_key == 'function':
            _create_change_patch = create_change_patch_with_function_context
        elif context_key == 'dependency':
            _create_change_patch = create_change_patch_with_dependency_context

        return Patch(
            id=patch_with_no_context.id,
            change_patch=_create_change_patch(patch_with_no_context),
            test_patch=patch_with_no_context.test_patch,
            path_to_root=patch_with_no_context.path_to_root,
        )

    def _collect_patches_with_no_context(
        self,
        patch_label: str,
        all_project_names: List[str],
        test_patch_fn: Callable
    ) -> List[Patch]:
        """Collects patches with no context for each hunk in the diff patch,
        i.e., when ``context_key`` is 'none'.

        Args:
            patch_label: Identifier for the type of patch to collect. For example,
                'gold' indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.
            all_project_names: List of all project names in benchmark
            test_patch_fn: Callable to create test patches. This needs to be defined
                independently for each `PatchCollector` object.

        Returns:
            patches: List of all patches in benchmark (created as ``Patch`` objects).
        """
        try:
            patches = self._load_patches('none', patch_label)
        except FileNotFoundError:
            # Extract all projects.
            all_projects = list((self.path_to_benchmark / 'projects').iterdir())
            patches = []

            for project in all_projects:
                if not project.stem in all_project_names:
                    continue
                # Extract all instances in a project.
                all_instances_in_project = list((project / 'instances').iterdir())
                for path_to_instance in all_instances_in_project:
                    if not path_to_instance.is_dir():
                        continue

                    with open(path_to_instance / 'gold_patch.txt', 'rb') as f:
                        patch_str = f.read().decode('utf-8')

                    patches.append(
                        self._create_patch_with_no_context(
                            patch_str,
                            path_to_instance,
                            test_patch_fn,
                        )
                    )
            # Save patches to cache.
            self._save_patches(patches, 'none', patch_label)
        return patches

    def _collect_patches_with_context(
        self,
        context_key: str,
        patch_label: str,
        all_project_names: List[str],
        test_patch_fn: Callable,
    ) -> List[Patch]:
        """Collects patches with context for each hunk in the diff patch,
        i.e., when ``context_key`` is one of 'function' or 'dependency'.

        To create patches with 'function' context, hunks with same diff
        header are grouped together, and the context is expanded till the
        containing function/class.

        To create patches with 'dependency' context, for the patches with
        'function' context, all caller-callee relationships are recorded.

        Args:
            context_key: Identifier for the amount of context to include for
                all chunks in a diff patch.
            patch_label: Identifier for the type of patch to collect. For example,
                'gold' indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.
            all_project_names: List of all project names in benchmark
            test_patch_fn: Callable to create test patches. This needs to be defined
                independently for each `PatchCollector` object.

        Returns:
            patches: List of all patches in benchmark (created as ``Patch`` objects).
        """
        try:
            patches = self._load_patches(context_key, patch_label)
            return patches

        except FileNotFoundError:
            if context_key == Context.DEPENDENCY:
                try:
                    patches_with_function_context = self._load_patches('function', patch_label)
                except FileNotFoundError:
                    patches_with_no_context = self._collect_patches_with_no_context(
                        patch_label,
                        all_project_names,
                        test_patch_fn,
                    )
                    patches_with_function_context = [self._create_patch_with_context(
                        patch_with_no_contex,
                        context_key
                        ) for patch_with_no_context in patches_with_no_context
                    ]
                    self._save_patches(
                        patches_with_function_context,
                        'function',
                        patch_label
                    )

                patches_with_dependency_context = [self._create_patch_with_context(
                    patch_with_function_context,
                    context_key
                    ) for patch_with_function_context in patches_with_function_context
                ]
                self._save_patches(
                    patches_with_dependency_context,
                    'dependency',
                    patch_label
                )
                return patches_with_dependency_context

            elif context_key == Context.FUNCTION:
                patches_with_no_context = self._collect_patches_with_no_context(
                    patch_label,
                    all_project_names,
                    test_patch_fn,
                )
                patches_with_function_context = [self._create_patch_with_context(
                    patch_with_no_context,
                    context_key,
                    ) for patch_with_no_context in patches_with_no_context
                ]
                self._save_patches(
                    patches_with_function_context,
                    'function',
                    patch_label
                )
                return patches_with_function_context

    def _collect(
        self,
        context_key: str,
        patch_label: str,
        all_project_names: List[str],
        test_patch_fn: Callable = None,
    ) -> List[Patch]:
        """Collect diff patches from corresponding benchmark repository.

        Args:
            context_key: Identifier for the amount of context to include for
                all chunks in a diff patch.
            patch_label: Identifier for the type of patch to collect. For example,
                'gold' indicates ground-truth patches, 'perturbation' indicates
                perturbed patches, etc.
            all_project_names: List of all project names in benchmark
            test_patch_fn: Callable to create test patches. This needs to be defined
                independently for each `PatchCollector` object.

        Returns:
            patches: List of all patches in benchmark (created as ``Patch`` objects).
        """
        self.path_to_cache.mkdir(parents=True, exist_ok=True)

        if context_key not in Context:
            raise ValueError(f"Invalid context parameter: {context_key}")

        if context_key == Context.NONE:
            return self._collect_patches_with_no_context(
                patch_label,
                all_project_names,
                test_patch_fn,
            )
        else:
            return self._collect_patches_with_context(
                context_key,
                patch_label,
                all_project_names,
                test_patch_fn,
            )

    def is_available(self, path: pathlib.Path) -> bool:
        """Check whether the original benchmark is available.
        
        Args:
            path: Path to benchmark.

        Returns:
            True if the benchmark is already available, False otherwise.
        """
        if path.is_dir():
            return True
        return False

    @abc.abstractmethod
    def collect(self) -> Any:
        """Process benchmark to collect data for agentic workflows."""
        pass
