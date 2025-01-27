import pathlib
import py_compile
import tempfile
from typing import Any, Dict

import jedi

from llm_agent_evaluation.data.patch_utils import Patch
from llm_agent_evaluation.scorers import BaseScorer


def get_compilation_status(source_code: str) -> bool:
    """Check if the source code is valid Python code.

    Args:
        source_code (str): Source code that needs to be checked.

    Returns:
        True, if the source code is valid Python code, False otherwise.
    """
    with tempfile.NamedTemporaryFile(suffix='.py') as temp_file:
        with open(temp_file.name, 'w') as f:
            f.write(source_code)
        try:
            py_compile.compile(temp_file.name, doraise=True)
            return True
        except py_compile.PyCompileError:
            return False


def get_number_of_syntax_errors(source_code: str) -> int:
    """Count the number of syntax errors in the source code.

    Args:
        source_code (str): Source code that needs to be checked.

    Returns:
        Number of syntax errors in the source code.
    """
    script = jedi.Script(code=source_code)
    number_errors = len(script.get_syntax_errors())
    return number_errors


class SyntacticScorer(BaseScorer):
    """A scorer that computes syntax-specific metrics for a patch."""
    _name = 'syntax'

    def __init__(
        self,
        benchmark: str,
        to_path: pathlib.Path
    ):
        """Initialize scorer for syntactic measures.

        Args:
            benchmark: Benchmark dataset name.
            to_path: Path to dataset resources.
        """
        self.logger = self.init_scoring_logger(benchmark, to_path)
        self.logger.info(
            ('Syntactic measures include: \n'
             '   (1) Validity, i.e., yes/no \n'
             '   (2) Number of syntax errors, e.g., 5 \n')
        )

    def score(self, input_patch: Patch) -> Dict[str, Dict[str, Any]]:
        """Score input patch on syntactic metrics.

        Args:
            input_patch: Input patch.

        Returns:
            Different syntactic measures and their values.
        """
        validity_scores, number_errors = [], []
        for chunk in input_patch.change_patch.after_chunks:
            # Each item in ``Chunk.lines`` is a tuple of <tag, line>.
            source_code = '\n'.join([line[1] for line in chunk.lines])
            validity_scores.append(get_compilation_status(source_code))
            number_errors.append(get_number_of_syntax_errors(source_code))
        return {
            'validity': validity_scores,
            'number_of_syntax_errors': number_errors,
        }
