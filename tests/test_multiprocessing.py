import pytest

import osekit.utils.multiprocess_utils as mpu
from osekit import config


class MockedPool:
    def __init__(self, processes: int):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def imap(self, func: callable, iterable: list):
        for element in iterable:
            yield func(element)


@pytest.mark.parametrize(
    ("multiprocessing", "nb_processes"),
    [
        pytest.param(
            False,
            None,
            id="multiprocessing_deactivated",
        ),
        pytest.param(
            False,
            10,
            id="multiprocessing_deactivated_shunts_nb_processes",
        ),
        pytest.param(
            True,
            None,
            id="multiprocessing_activated",
        ),
        pytest.param(
            True,
            10,
            id="multiprocessing_activated_with_specified_nb_processes",
        ),
    ],
)
def test_multiprocessing(
    monkeypatch: pytest.MonkeyPatch,
    multiprocessing: bool,
    nb_processes: int,
) -> None:
    monkeypatch.setitem(config.multiprocessing, "is_active", multiprocessing)
    monkeypatch.setitem(config.multiprocessing, "nb_processes", nb_processes)

    def add(x: int, y: int) -> int:
        return x + y

    pool_call = {"called": False, "nb_processes": None}

    def patch_pool(nb_processes: int):
        pool_call["called"] = True
        pool_call["nb_processes"] = nb_processes
        return MockedPool(nb_processes)

    monkeypatch.setattr(mpu.mp, "Pool", patch_pool)

    tqdm_call = {"called": False, "disable": False}

    def patch_tqdm(iterable: list[int], disable: bool, total: int = 0) -> list[int]:
        tqdm_call["called"] = True
        tqdm_call["disable"] = disable
        return iterable

    monkeypatch.setattr(mpu, "tqdm", patch_tqdm)

    result = mpu.multiprocess(add, [1, 2, 3], y=5)

    assert result == [6, 7, 8]
    assert tqdm_call["called"] is True
    assert pool_call["called"] is multiprocessing
    assert pool_call["nb_processes"] == (nb_processes if multiprocessing else None)
