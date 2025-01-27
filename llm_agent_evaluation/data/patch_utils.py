import difflib
import enum
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from collections import namedtuple
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Union

from git import Repo

from llm_agent_evaluation.utils import CustomEnumMeta
from llm_agent_evaluation.external.ast.explorer import (
    build_ast_from_source,
    find_nodes_of_type,
)


class Chunk:
    """Data structure representing different hunks in a diff patch. When
    within a function in a diff patch, we refer to a 'hunk' as a 'chunk`.
    At the function/class-level context, a group of hunks is a chunk.

    Note that `Chunk` could represent both the before and after versions.

    Attributes:
        start_lineno: Starting line number of a hunk in containing file. When
            merging all hunks for function/class-level context, this refers to
            the starting line number of the containing function/class.
        filename: Name of the containing file.
        header: Header of the chunk in the format "@@ -s,n +s,n @@"
        lines: List of tuples containing a tag, and line contents. Here, tag
            can be '0' (i.e., unmodified) or '1' (i.e., added/deleted).
    """
    def __init__(
        self,
        start_lineno: int,
        filename: str,
        header: str,
        lines: List[Tuple[int, str]]
    ):
        self.start_lineno = start_lineno
        self.filename = filename
        self.header = header
        self.lines = lines


class ChangePatch:
    """Data structure representing a diff change patch.
    
    Attributes:
        text: Code patch string in diff patch.
        before_chunks: List of `Chunk` objects before code change.
        after_chunks: List of `Chunk` objects after code change.
    """
    def __init__(
        self,
        text: str,
        before_chunks: List[Chunk],
        after_chunks: List[Chunk],
    ):
        self.text = text
        self.before_chunks = before_chunks
        self.after_chunks = after_chunks


class TestPatch:
    """Data structure representing a diff test patch.
    
    Attribute:
        text: Text patch string in diff patch.
        relevant_tests: Relevant unit tests (e.g., buggy unit tests).
    """
    def __init__(self, text: str, relevant_tests: str):
        self.text = text
        self.relevant_tests = relevant_tests


class Patch:
    """Data structure representing a diff patch.

    Attributes:
        id: Identifier (e.g., <project>-00<bug-id> in BugsInPy).
        change_patch: Code change patch.
        test_patch: Test change patch.
        path_to_root: Path to the root of the code repository containing
            all repository snapshots.
    """
    def __init__(
        self,
        id: str,
        change_patch: ChangePatch,
        test_patch: TestPatch,
        path_to_root: pathlib.Path
    ):
        self.id = id
        self.change_patch = change_patch
        self.test_patch = test_patch
        self.path_to_root = path_to_root

    def __str__(self):
        return pretty_print_patch(self)


class Context(str, enum.Enum, metaclass=CustomEnumMeta):
    """Amount of context for a chunk in a patch."""
    NONE = 'none'
    FUNCTION = 'function'
    DEPENDENCY = 'dependency'


def extract_change_and_test_patches(patch: str) -> Tuple[str, str]:
    """Get change and test patches from a diff patch.

    Source:
    https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/collect/utils.py

    Args:
        patch: Diff patch.
    
    Returns:
        change_text, test_text:  Change and test patch strings.
    """
    if patch.endswith("\n"):
        patch = patch[:-1]

    # Create change patch and test patch
    patch_change, patch_test = [], []

    # Flag to determine if current diff block is a test or general change
    # Values: 'test', 'diff', None
    flag = None

    for line in patch.split("\n"):
        # In swe-bench, commit specific metadata is omitted. We retain it.
        if line.startswith("index "):
            patch_change.append(line)

        # Determine if current diff block is a test or general change
        if line.startswith("diff --git a/"):
            words = set(re.split(r" |_|\/|\.", line.lower()))
            flag = (
                "test"
                if ("test" in words or "tests" in words or "testing" in words)
                else "diff"
            )
        
        # Append line to separate patch depending on flag status
        if flag == "test":
            patch_test.append(line)
        elif flag == "diff":
            patch_change.append(line)

    change_text = ""
    if patch_change:
        change_text = "\n".join(patch_change) + "\n"

    test_text = ""
    if patch_test:
        test_text = "\n".join(patch_test) + "\n"

    return change_text, test_text


def extract_fine_grained_changes(change_text) -> dict:
    """Extracts a fine-grained analysis of the change text, grouping all
    hunks into a diff group.

    Args:
        change_text: Change patch string.

    Returns:
        Diff groups (i.e., hunks in patch).
    """
    lines = change_text.split('\n')
    diff_groups = []

    # Diff group level
    current_group = []
    for line in lines:
        if (line.startswith('+') or line.startswith('-')):
            current_group.append(line)
        else:
            if current_group:
                diff_groups.append(current_group)
                current_group = []

    # Add the last group if it exists
    if current_group:
        diff_groups.append(current_group)

    return diff_groups


def parse_git_diff_enhanced(change_text: str) -> List[Dict]:
    """Utility to parse git diff and extract all components.

    Args:
        change_text (str): Change patch string in diff patch.

    Returns:
        parsed_diff: A dictionary of all items in the diff. It has the following
            keys 'before_index' 'after_index', 'mode', 'filename', and 'changes'. 
    
        Here, 'changes' is a list of dictionaries representing each change
        in a file, with the following keys: 'text' (hunk segment with context),
        'before_start_line' (line in source file before revision), 'before_count'
        (number of lines in hunk before revision), 'after_start_line' (line in
        source file after revision), 'after_count' (number of lines in hunk
        after revision), and 'diff_groups' (a list of fine-grained changes in
        the hunk). Each fine-grained change is a dictionary with the following
    """
    # Regex patterns to capture different components of the diff
    file_pattern = r"diff --git a/(.+) b/\1"
    index_pattern = r"index (\w+)\.\.(\w+) (\d+)"
    header_pattern = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@\s+(.*?)(?=(^@@ -)|\Z)"

    # Split the diff text into file sections
    file_sections = re.split(file_pattern, change_text)[1:]

    parsed_diff = []
    for i in range(0, len(file_sections), 2):
        file_diff = {}
        file_name = file_sections[i].strip()
        file_content = file_sections[i+1]

        # Extracting index information
        index_match = re.search(index_pattern, file_content)
        if index_match:
            file_diff['before_index'] = index_match.group(1)
            file_diff['after_index'] = index_match.group(2)
            file_diff['mode'] = index_match.group(3)

        # Extracting changes
        changes = re.findall(header_pattern, file_content, re.DOTALL | re.MULTILINE)

        formatted_changes = []
        for change in changes:
            change_dict = {
                'text': '\n'.join(change[4].split('\n')),
                'before_start_line': int(change[0]),
                'before_count': int(change[1]),
                'after_start_line': int(change[2]),
                'after_count': int(change[3]),
                'diff_groups': extract_fine_grained_changes(change[4])
            }
            formatted_changes.append(change_dict)

        file_diff['filename'] = file_name
        file_diff['changes'] = formatted_changes
        parsed_diff.append(file_diff)

    return parsed_diff


def correct_modified_lines_offset(
    start_lineno: int,
    modified_lines_with_context: List[str],
    modified_lines: List[str],
) -> int:
    """The ``start_lineno`` attribute in ``Chunk`` object is expected to
    correspond to the first modified line in a hunk (with respect to the
    containing source file). To extract this attribute, we utilize the
    diff header and the hunk contents.

    Note that in the case of function/class-level context, this simply
    represents the starting line number oif the function/class.

    Args:
        start_lineno: Starting line number in diff header of a hunk.
        modified_lines_with_context: List of modified lines in a hunk
            with +-k lines context, as in a patch file.
        modified_lines: List of modified lines in a hunk.

    Returns:
        Corrected starting line number of a hunk in containing file.
    """
    for idx in range(len(modified_lines_with_context) - len(modified_lines) + 1):
        if modified_lines_with_context[idx: idx+len(modified_lines)] == modified_lines:
            return start_lineno + idx


def group_chunks_by_attribute(
    all_chunks: List[Chunk],
    attr: str
) -> Dict[str, List[Chunk]]:
    """A chunk can be grouped based on all belonging to the same file
    (i.e., having the same 'filename'), or all belonging to the same
    function/class (i.e., having the same 'header').

    Here, we group based on the selected ``attr`` attribute.

    Args:
        all_chunks: List of chunks (before or after revision).
        attr: Attribute to group the chunks by.

    Returns:
        grouped_chunks: Dictionary mapping groups of chunks to their attributes.
    """
    grouped_chunks = {}
    for chunk in all_chunks:
        attr_value = getattr(chunk, attr)
        if attr_value in grouped_chunks:
            grouped_chunks[attr_value].append(chunk)
        else:
            grouped_chunks[attr_value] = [chunk]
    return grouped_chunks


def merge_chunks_in_header(
    all_chunks: List[Chunk],
    path_to_repo: pathlib.Path
) -> List[Chunk]:
    """Merge group of chunks within a header, i.e., within the same function/class
    to create a single chunk representing the entire function/class-level context.

    Args:
        all_chunks: List of chunks (before or after revision).
        path_to_repo: Path to the code repository snapshot.

    Returns:
        merged_chunks: Merged chunks (each chunk has a unique <file_name, header> group).
    """
    merged_chunks = []
    for file_name, grouped_chunks_by_filename in group_chunks_by_attribute(
        all_chunks, 'filename'
    ).items():
        # Ignore non-Python related files in diff patch.
        if not str(file_name).endswith('.py'):
            continue

        with open(str(path_to_repo / file_name), 'r') as f:
            source = f.read()
        source_lines = source.split('\n')

        tree = build_ast_from_source(source)
        def_nodes = list(find_nodes_of_type(tree, 'class_definition')) + \
                    list(find_nodes_of_type(tree, 'function_definition'))

        _merged_chunks = group_chunks_by_attribute(
            grouped_chunks_by_filename, 'header'
        )

        for header, _chunks in _merged_chunks.items():
            containing_node = None
            for node in def_nodes:
                definition_text = source_lines[node.start_point[0]].strip()
                if header.strip() in definition_text:
                    containing_node = node
                    break

            # Consider hunks only in function/class-level context.
            if not containing_node:
                continue

            start_lineno = containing_node.start_point[0]
            end_lineno = containing_node.end_point[0]
            containing_code = source_lines[start_lineno: end_lineno+1]

            all_tags = [0 for _ in range(len(source_lines))]

            for chunk in _chunks:
                for idx in range(chunk.start_lineno, chunk.start_lineno+len(chunk.lines)):
                    # Note that chunk.start_lineno is one-indexed and not zero-indexed.
                    # Correcting the offset here with ``idx-1``.
                    all_tags[idx-1] = 1
            tags = all_tags[start_lineno: end_lineno+1]

            merged_chunks.append(
                Chunk(
                    start_lineno=containing_node.start_point[0],
                    filename=file_name,
                    header=header,
                    lines=[(tag, containing_code[idx]) for idx, tag in enumerate(tags)],
                )
            )
    return merged_chunks


def create_change_patch_with_no_context(change_text: str) -> ChangePatch:
    """Extract all chunks in a change patch with no additional context.
    By default, diff patch carries +-3 lines around a hunk.

    Args:
        change_text: Change patch string in diff patch.

    Returns:
        ``ChangePatch`` object containing all chunks with no context.
    """
    # Parse diff and extract all chunks.
    parsed_diff = parse_git_diff_enhanced(change_text)
    before_chunks, after_chunks = [], []
    for item in parsed_diff:
        for i, change in enumerate(item['changes']):
            change_lines = change['text'].split('\n')
            for hunk in change['diff_groups']:
                removed_lines = [line for line in hunk if line.startswith('-')]
                removed_lines_with_context = [
                    line for line in change_lines[1:] if not line.startswith('+')
                ]
                if removed_lines:
                    before_chunks.append(
                        Chunk(
                            start_lineno=correct_modified_lines_offset(
                                change['before_start_line'],
                                removed_lines_with_context,
                                removed_lines,
                            ),
                            filename=item['filename'],
                            header=change_lines[0],
                            lines=[(1, line[1:]) for line in removed_lines],
                        )
                    )

                added_lines = [line for line in hunk if line.startswith('+')]
                added_lines_with_context = [
                    line for line in change_lines[1:] if not line.startswith('-')
                ]

                if added_lines:
                    after_chunks.append(
                        Chunk(
                            start_lineno=correct_modified_lines_offset(
                                change['after_start_line'],
                                added_lines_with_context,
                                added_lines,
                            ),
                            filename=item['filename'],
                            header=change_lines[0],
                            lines=[(1, line[1:]) for line in added_lines],
                        )
                    )

    return ChangePatch(
        text=change_text,
        before_chunks=before_chunks,
        after_chunks=after_chunks,
    )


def create_change_patch_with_function_context(
    patch: Patch,
    patch_label: str = 'gold',
) -> ChangePatch:
    """Extract all chunks in a patch with function context, i.e., all
    hunks in a patch are expanded till the containing function/class.

    Args:
        patch: ``Patch`` object with no additional context.
        patch_label: Label for type of patch. In the case of ground-truth
            patches, this'd be 'gold'.

    Returns:
        ``ChangePatch`` object containing all chunks with function-context.
    """
    return ChangePatch(
        text=patch.change_patch.text,
        before_chunks=merge_chunks_in_header(
            patch.change_patch.before_chunks,
            patch.path_to_root / 'snapshots/base',
        ),
        after_chunks=merge_chunks_in_header(
            patch.change_patch.after_chunks,
            patch.path_to_root / f'snapshots/{patch_label}',
        ),
    )


def create_change_patch_with_dependency_context(
    change_text: str,
    patch_label: str = 'gold',
    ) -> ChangePatch:
    """Extract all chunks in a patch with dependency context, i.e., by
    including all 'calls' and `calledBy` functions in its static call graph.

    Args:
        change_text: Change patch string in diff patch.
        patch_label: Label for type of patch. In the case of ground-truth
            patches, this'd be 'gold'.

    Returns:
        ``ChangePatch`` object containing all chunks with function-context.
    """
    pass


def pretty_print_patch(patch: Patch) -> str:
    """Print a readable-version of a patch.

    Args:
        patch: Input patch to be pretty-printed

    Returns:
        output: String representation of the patch in a pretty printed format.
    """
    output = [
        '='*10 + f' Diff patch from {patch.path_to_root} ' + '='*10,
        '-'*10 + ' Change patch ' + '-'*10,
    ]
    output += patch.change_patch.text.split('\n')
    output.append('')
    if patch.change_patch.before_chunks:
        for idx, chunk in enumerate(patch.change_patch.before_chunks):
            output.append('.'*10 + f' Before revision, chunk {idx+1} ' + '.'*10)
            output.append(f'  start lineno: {chunk.start_lineno}')
            output.append(f'  filename: {chunk.filename}')
            output.append(f'  header: {chunk.header}')
            for (tag, line) in chunk.lines:
                label = '-' if tag == 1 else ' '
                output.append(f'{tag}\t{line}')
        output.append('\n')

    if patch.change_patch.after_chunks:
        for idx, chunk in enumerate(patch.change_patch.after_chunks):
            output.append('.'*10 + f' After revision, chunk {idx+1} ' + '.'*10)
            output.append(f'  start lineno: {chunk.start_lineno}')
            output.append(f'  filename: {chunk.filename}')
            output.append(f'  header: {chunk.header}')
            for (tag, line) in chunk.lines:
                label = '+' if tag == 1 else ' '
                output.append(f'{tag}\t{line}')

    if patch.test_patch.relevant_tests:
        output.append('-'*10 + ' Test patch ' + '-'*10)
        for idx, unit_test in enumerate(patch.test_patch.relevant_tests):
            output.append('.'*10 + f' Relevant unit test {idx+1} ' + '.'*10)
            output += unit_test.split('\n')

    output = '\n'.join(output)
    return output


def apply_patch(
    patch: str,
    path_to_root: pathlib.Path,
    identifier: str,
    change_only: bool = True,
) -> None:
    """Apply patch to a repo snapshot. A new repo snapshot of the form
    ``<path-to-repo>/snapshots/new-<identifier>`` is created with all
    changes in ``patch`` applied.

    Args:
        patch: The patch string to apply.
        path_to_root: Path to the root of the code repository containing both
            the 'old' and 'new' version snapshots.
        identifier: The identifier for the new snapshot.
        change_only: Whether to apply only the change part of the patch, or the
            test part of the patch as well. True, by default.
    """
    change_patch, test_patch = extract_change_and_test_patches(patch)

    patch_to_apply = patch
    if change_only:
        patch_to_apply = change_patch

    shutil.copytree(str(path_to_root / 'old'), str(path_to_root / f'new_{identifier}'))

    with tempfile.NamedTemporaryFile(suffix='.diff') as temp_file:
        with open(temp_file.name, 'w') as f:
            f.write(patch_to_apply)

        repo = Repo(str(path_to_root / f'new_{identifier}'))
        repo.git.execute(['git', 'apply', str(temp_file.name)])


def apply_patch_to_repo(
    patch: str,
    repo_dir: str,
    logger: logging.getLogger,
) -> Union[Set[str], bool]:
    """Applies a unified diff patch to a git repository and returns the affected files.

    Source: This is a copy of Tim Esler's implementation.

    Args:
        patch: The unified diff patch as a string.
        repo_dir: The directory of the git repository.
        logger: Logger.

    Returns:
        A set of modified files if the patch is applied successfully, False otherwise.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_patch_file:
            temp_patch_file.write(patch.encode())
            temp_patch_file_path = temp_patch_file.name

        try:
            subprocess.run(
                ["git", "stash"],
                cwd=repo_dir,
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "apply", temp_patch_file_path],
                cwd=repo_dir,
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )

            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            modified_files = set(result.stdout.splitlines())

            # Get git diff with additional context lines
            result = subprocess.run(
                ["git", "diff", "-U10"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            patch_with_additional_context = result.stdout
            return modified_files, patch_with_additional_context

        except subprocess.CalledProcessError as e:
            logger.debug(f"Error applying patch: {e.stderr.decode()}")
            return False, ""

        finally:
            os.remove(temp_patch_file_path)
    except:
        pass