import pathlib
from typing import Dict, List, Tuple

from radon.complexity import cc_visit, average_complexity
from radon.metrics import h_visit

from llm_agent_evaluation.data.patch_utils import (
    Chunk,
    Patch,
    group_chunks_by_attribute
)
from llm_agent_evaluation.scorers import BaseScorer


def extract_chunk_pairs(
    before_chunks: List[Chunk],
    after_chunks: List[Chunk],
) -> Tuple[Chunk, Chunk]:
    """Extract pairs of before and after versions of a code chunk.
    
    Args:
        before_chunks: List of before versions of code chunks.
        after_chunks: List of after versions of code chunks.

    Returns:
        Pairs of before and after code chunks.
    """
    return [
        (bchunk, achunk)
        for bchunk in before_chunks
        for achunk in after_chunks
        if bchunk.filename == achunk.filename and bchunk.header == achunk.header
    ]


def get_cyclomatic_complexity(code: str) -> float:
    """Compute cyclomatic complexity of a code snippet.

    Args:
        code: Code snippet.

    Returns:
        Cyclomatic complexity of the code snippet. None, if there is an
        error in computing the complexity.
    """
    try:
        return average_complexity(cc_visit(code))
    except:
        return None


def get_halstead_metric(code: str) -> float:
    """Compute Halstead metrics (difficulty) of a code snippet.

    Args:
        code: Code snippet.

    Returns:
        Halstead metrics of the code snippet. None, if there is an error
        in computing the metrics.
    """
    try:
        return h_visit(code).total.difficulty
    except:
        return None


class ComplexityScorer(BaseScorer):
    """A scorer that computes complexity metrics for both before and after
    versions of a patch.
    """
    _name = 'complexity'

    def __init__(self, benchmark: str, to_path: pathlib.Path):
        logger = self.init_scoring_logger(benchmark, to_path)
        logger.info(
            ('Complexity metrics include: \n'
             '   (1) Cyclomatic complexity \n'
             '   (2) Halstead metrics (length, vocabulary, volume, difficulty, effort) \n')
        )

    def score(self, input_patch: Patch) -> Dict:
        """Score input patch on complexity metrics.

        Args:
            input_patch: Input patch.

        Returns:
            Different complexity measures and their values.
        """
        cyclomatic, halstead = [], []
        for (before_chunk, after_chunk) in extract_chunk_pairs(
            input_patch.change_patch.before_chunks,
            input_patch.change_patch.after_chunks,
        ):
            before_code = '\n'.join([line[1] for line in before_chunk.lines])
            after_code = '\n'.join([line[1] for line in after_chunk.lines])

            # If both `before_code` and `after_code` are parseable and there
            # are no errors in computing Cyclomatic complexity or Halstead
            # metrics, these are computed. Else, `None` is returned.
            cyclomatic.append(
                (
                    get_cyclomatic_complexity(before_code),
                    get_cyclomatic_complexity(after_code),
                )
            )            

            halstead.append(
                (
                    get_halstead_metric(before_code),
                    get_halstead_metric(after_code),
                )
            )
        return {
            'Cyclomatic complexity': cyclomatic,
            'Halstead metrics': halstead,
        }
