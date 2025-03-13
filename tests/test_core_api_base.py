from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pandas import Timestamp

from OSmOSE.core_api.base_file import BaseFile

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "destination_folder",
    [
        pytest.param(
            "cool",
            id="moving_to_new_folder",
        ),
        pytest.param(
            "",
            id="moving_to_same_folder",
        ),
    ],
)
def test_move_file(
    tmp_path: Path,
    destination_folder: str,
) -> None:
    filename = "cool.txt"
    (tmp_path / filename).touch(mode=0o666, exist_ok=True)
    bf = BaseFile(
        tmp_path / filename,
        begin=Timestamp("2022-04-22 12:12:12"),
        end=Timestamp("2022-04-22 12:13:12"),
    )

    bf.move(tmp_path / destination_folder)

    assert (tmp_path / destination_folder / filename).exists()

    if destination_folder:
        assert not (tmp_path / filename).exists()
