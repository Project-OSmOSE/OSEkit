"""Multiprocessing module that helps running functions on collections using multiple threads."""

import multiprocessing as mp
import os
from functools import partial
from typing import Any

from tqdm import tqdm

from osekit import config


def multiprocess(
    func: callable,
    enumerable: list,
    *args: Any,
    bypass_multiprocessing: bool = False,
    **kwargs: Any,
) -> list[Any]:
    """Run a given callable function on an enumerable.

    The function is run through ``osekit.config.nb_processes`` threads.

    Parameters
    ----------
    func: callable
        The function to run.
    enumerable: list
        The list of input to the function.
    bypass_multiprocessing: bool
        If ``True``, multiprocessing will be bypassed whatever the config value.
    args:
        Additional positional arguments to pass to the function.
    kwargs:
        Additional keyword arguments to pass to the function.

    Returns
    -------
    list[Any]:
        Returned values of the function.

    """
    if bypass_multiprocessing or not config.multiprocessing["is_active"]:
        return list(
            func(element, *args, **kwargs)
            for element in tqdm(
                enumerable,
                disable=os.getenv("DISABLE_TQDM", "False").lower()
                in ("true", "1", "t"),
            )
        )

    partial_func = partial(func, *args, **kwargs)

    with mp.Pool(config.multiprocessing["nb_processes"]) as pool:
        return list(
            tqdm(
                pool.imap(partial_func, enumerable),
                total=len(list(enumerable)),
                disable=os.getenv("DISABLE_TQDM", "False").lower()
                in ("true", "1", "t"),
            ),
        )
