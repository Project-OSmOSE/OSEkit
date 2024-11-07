from __future__ import annotations

import os
from pathlib import Path
import logging
import pytest

import OSmOSE
from OSmOSE.utils.core_utils import chmod_if_needed, chown_if_needed


@pytest.mark.unittest
def test_no_error_o_non_unix_os(tmp_path: Path) -> None:

    OSmOSE.utils.core_utils._is_grp_supported = False
    try:
        chown_if_needed(path=tmp_path, owner_group="test")
        chmod_if_needed(path=tmp_path, mode=0o775)
    except Exception as e:
        pytest.fail(str(e))

@pytest.mark.unittest
def test_no_chmod_attempt_if_not_needed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:

    OSmOSE.utils.core_utils._is_grp_supported = True

    monkeypatch.setattr(os, "access", lambda path,mode: mode in [os.R_OK, os.W_OK])
    monkeypatch.setattr(Path, "chmod", lambda path, mode: pytest.fail("Call to chmod although user already has read and write permissions."))

    chmod_if_needed(path=tmp_path, mode=0o775)

@pytest.mark.unittest
@pytest.mark.parametrize(
    ("new_mode", "path_access"),
    [
        pytest.param(
            0o775,
            [os.R_OK],
            id="missing_write_permission",
        ),
        pytest.param(
            0o775,
            [os.W_OK],
            id="missing_read_permission",
        ),
        pytest.param(
            0o775,
            [],
            id="missing_all_permission",
        ),
    ],
)
def test_chmod_called_if_missing_permissions(new_mode: int, path_access: list[int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:

    OSmOSE.utils.core_utils._is_grp_supported = True

    file_mode = 0o664

    def set_mode(mode: int) -> None:
        nonlocal file_mode
        file_mode = mode

    monkeypatch.setattr(os, "access", lambda path,mode: mode in [os.R_OK])
    monkeypatch.setattr(Path, "chmod", lambda path, mode: set_mode(mode))

    chmod_if_needed(path=tmp_path, mode=new_mode)
    assert file_mode==new_mode


def test_error_logged_if_no_chmod_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: logging.LogCaptureFixture) -> None:

    OSmOSE.utils.core_utils._is_grp_supported = True
    monkeypatch.setattr(os, "access", lambda path, mode: mode in [])

    def raise_permission_error() -> None:
        raise PermissionError

    monkeypatch.setattr(Path, "chmod", lambda path, mode: raise_permission_error())

    with caplog.at_level(logging.WARNING), pytest.raises(PermissionError) as e:
        assert chmod_if_needed(path=tmp_path, mode=0o775) == e

    assert f"You do not have the permission to write to {tmp_path}" in caplog.text