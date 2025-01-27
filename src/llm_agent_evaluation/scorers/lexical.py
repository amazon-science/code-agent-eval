import enum
import itertools
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

from scipy.spatial.distance import cosine

from simpletransformers.language_representation import RepresentationModel

import torch

from llm_agent_evaluation.data.patch_utils import Patch
from llm_agent_evaluation.scorers import BaseScorer
from llm_agent_evaluation.scorers.scorer_utils import (
    extract_flattened_change_patch
)
from llm_agent_evaluation.utils import CustomEnumMeta


SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class EditMeasure(str, enum.Enum, metaclass=CustomEnumMeta):
    """Type of edit distance, can be text-based or embedding-based."""
    TEXT = 'text'
    EMBEDDING = 'embedding'


def levenshtein_distance(s1, s2):
    """Computes Levenshtein distance between two code strings.

    Reference:
    https://stackoverflow.com/questions/2460177/edit-distance-in-python

    Args:
        s1: First code string
        s2: Second code string

    Returns:
        The Levenshtein distance between the two code strings.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def cosine_similarity(s1, s2):
    """Computes cosine similarity between two code strings, representations
    of which are obtained from CodeBERT model.

    Args:
        s1: First code string
        s2: Second code string

    Returns:
        Cosine similarity between representations of two code strings.
    """
    model = RepresentationModel(
                "roberta",
                "microsoft/codebert-base",
                use_cuda=torch.cuda.is_available(),
    )
    # Compute code string embeddings with CodeBERT model.
    embeddings = model.encode_sentences(
        [s1, s2],
        combine_strategy='mean'
    )
    return cosine(embeddings[0], embeddings[1])


def edit_distance(
    code_string1: str,
    code_string2: str,
    measure: str = 'text'
) -> float:
    """Computes the edit distance between two code strings.

    Args:
        code_string1: First code string.
        code_string2: Second code string.
        measure: The measure to use. Can be either 'text' or 'embedding'.

    Returns:
        The edit distance between the two code strings.
    """
    if measure not in EditMeasure:
        raise ValueError(f"Invalid edit measure parameter: {measure}")

    if measure == EditMeasure.TEXT:
        edit_distance_fn = levenshtein_distance
    elif measure == EditMeasure.EMBEDDING:
        edit_distance_fn = cosine_similarity

    return edit_distance_fn(code_string1, code_string2)


def get_number_of_changed_lines(patch: Patch) -> int:
    """Returns the number of changed lines in a patch.

    Args:
        patch: Patch object.

    Returns:
        The number of changed lines in the change patch in a ``Patch`` object.
    """
    # Each item in ``Chunk.lines`` is a tuple of <tag, line>, where 'tag' is 1 or 0,
    # depending on whether the line is modified or not, respectively. Here, we collect
    # the total number of such modified lines before and after the revision.
    #
    # Note that this is different from the total number of modified lines in a raw diff
    # patch, because, in this project, we only consider modified lines within a function
    # or class in a .py file as a `Chunk` object. As a result, modifications to non-Python
    # files, or in global variables, import statements, etc. will be ignored.
    number_of_changed_lines = sum(
        item[0] == 1
        for chunk in patch.change_patch.before_chunks + patch.change_patch.after_chunks
        for item in chunk.lines
    )
    return number_of_changed_lines


def get_number_of_changed_files(patch: Patch) -> Tuple[int, int]:
    """Returns the number of changed files in a patch.

    Args:
        patch: Patch object.

    Returns:
        The number of changed files in the change patch in a ``Patch`` object.
    """
    all_files = {
        chunk.filename 
        for chunks in [patch.change_patch.before_chunks, patch.change_patch.after_chunks]
        for chunk in chunks
    }
    return len(all_files)


class LexicalScorer(BaseScorer):
    """A scorer that computes lexical metrics for a patch."""
    _name = 'lexical'

    def __init__(
        self,
        benchmark: str,
        to_path: pathlib.Path
    ):
        """Initialize scorer for lexical metrics.

        Args:
            benchmark: Benchmark dataset name.
            to_path: Path to dataset resources.
        """
        self.logger = self.init_scoring_logger(benchmark, to_path)
        self.logger.info(
            ('Lexical metrics include: \n'
             '   (1) Number of changed lines \n'
             '   (2) Number of changed files \n'
             '   (3) Edit distance between before and after versions (text-based) \n'
             '   (4) Edit distance between before and after versions (embedding-based) \n')
        )

    def score(self, input_patch: Patch, gold_patch: Patch = None) -> Dict[str, Any]:
        """Score input patch on lexical metrics.

        Args:
            input_patch: Input patch.
            gold_patch: Ground-truth patch.

        Returns:
            Different lexical measures and their values.
        """
        # Number of changed lines, and files, are reference-free metrics.
        number_of_changed_lines = get_number_of_changed_lines(input_patch)
        number_of_changed_files = get_number_of_changed_files(input_patch)

        if not gold_patch:
            return {
                'number_of_changed_lines': number_of_changed_lines,
                'number_of_changed_files': number_of_changed_files,
            }

        return {
            'number_of_changed_lines': number_of_changed_lines,
            'number_of_changed_files': number_of_changed_files,
            'edit_text': edit_distance(
                gold_patch.change_patch.text,
                input_patch.change_patch.text,
                'text',
            ),
            'edit_embedding': edit_distance(
                gold_patch.change_patch.text,
                input_patch.change_patch.text,
                'embedding',
            ),
        }
