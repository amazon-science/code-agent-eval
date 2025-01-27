import abc
import logging
import pathlib


class BaseScorer(abc.ABC):
    """Abstract base class for scoring source code on proxies, such as,
    text-based, complexity, syntax, semantics, and execution.
    """
    _name = None

    @property
    def name(self) -> str:
        """Return the name of the scorer."""
        return self._name

    def init_scoring_logger(
        self,
        benchmark: str,
        to_path: pathlib.Path
    ) -> logging.Logger:
        """Initialize a logger that saves all benchmark-specific source code
        scoring logs to its 'score_logs_{lexical|syntax|semantics|execution}.txt' file.

        Args:
            benchmark: Name of the benchmark.
            to_path: Path-like object to all resources.

        Returns:
            logger (logging.Logger)
        """
        path_to_score_logs = to_path / 'logs' / benchmark
        path_to_score_logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(path_to_score_logs / f'score_logs_{self.name}.txt'),
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )
        logger = logging.getLogger(__name__)
        return logger

    @abc.abstractmethod
    def score(self) -> None:
        """Score a patch."""
        pass
