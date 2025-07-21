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
    **kwargs: Any,
) -> list[Any]:
    """Run a given callable function on an enumerable.

    The function is run through osekit.config.nb_processes threads.

    Parameters
    ----------
    func: callable
        The function to run.
    enumerable: list
        The list of input to the function.
    args:
        Additional positional arguments to pass to the function.
    kwargs:
        Additional keyword arguments to pass to the function.

    Returns
    -------
    list[Any]:
        Returned values of the function.

    """
    if not config.multiprocessing["is_active"]:
        return list(
            func(element, *args, **kwargs)
            for element in tqdm(enumerable, disable=os.environ.get("DISABLE_TQDM", ""))
        )

    partial_func = partial(func, *args, **kwargs)

    with mp.Pool(config.multiprocessing["nb_processes"]) as pool:
        return list(
            tqdm(
                pool.imap(partial_func, enumerable),
                total=len(list(enumerable)),
                disable=os.environ.get("DISABLE_TQDM", ""),
            ),
        )
