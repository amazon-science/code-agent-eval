import logging
import pathlib


class IncompleteProjectInfoError(Exception):
    """Exception to throw if all project-specific information is not
    available in the corresponding 'project.info' file.
    """
    def __init__(self, project_path: pathlib.Path, logger: logging.Logger):
        self.msg = f'Incomplete project information in {project_path}.'
        logger.error(self.msg)

    def __str__(self):
        return self.msg


class InvalidProjectError(Exception):
    """Exception to throw if project status is set to 'NOT OK' in its
    'project.info' file.
    """
    def __init__(self, project_name: str, logger: logging.Logger):
        self.msg = f'Project {project_name} status is not OK.'
        logger.error(self.msg)

    def __str__(self):
        return self.msg


class IncompleteBugInfoError(Exception):
    """Exception to throw if all bug-specific information is not available
    in the corresponding 'bug.info' file.
    """
    def __init__(self, bug_path: pathlib.Path, logger: logging.Logger):
        self.msg = f'Incomplete bug information in {bug_path}.'
        logger.error(self.msg)

    def __str__(self):
        return self.msg
