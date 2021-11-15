import os
from typing import Optional

from clozetask.core import Task
from clozetask.task import SemevalTask
from clozetask.utils.io import read_json

TASK_DICT = {
    "classTask": SemevalTask
}

def get_task_class(task_name: str):
    task_class = TASK_DICT[task_name]
    assert issubclass(task_class, Task)
    return task_class


def create_task_from_config(config: dict, base_path: Optional[str] = None, verbose: bool = False):
    """Create task instance from task config.
    Args:
        config (Dict): task config map.
        base_path (str): if the path is not absolute, path is assumed to be relative to base_path.
        verbose (bool): True if task config should be printed during task creation.
    Returns:
        Task instance.
    """
    task_class = get_task_class(config["task"])
    for k in config["paths"].keys():
        path = config["paths"][k]
        # TODO: Refactor paths  (issue #1180)
        if isinstance(path, str) and not os.path.isabs(path):
            assert base_path
            config["paths"][k] = os.path.join(base_path, path)
    task_kwargs = config.get("kwargs", {})
    if verbose:
        print(task_class.__name__)
        for k, v in config["paths"].items():
            print(f"  [{k}]: {v}")
    # noinspection PyArgumentList
    return task_class(name=config["name"], path_dict=config["paths"], **task_kwargs)


def create_task_from_config_path(config_path: str, verbose: bool = False):
    """Creates task instance from task config filepath.
    Args:
        config_path (str): config filepath.
        verbose (bool): True if task config should be printed during task creation.
    Returns:
        Task instance.
    """
    return create_task_from_config(
        read_json(config_path), base_path=os.path.split(config_path)[0], verbose=verbose,
    )

