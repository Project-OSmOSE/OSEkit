from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import osekit
from osekit.utils.core_utils import change_owner_group, chmod_if_needed

if TYPE_CHECKING:
    from unittest.mock import MagicMock


@pytest.mark.unit
def test_no_error_o_non_unix_os(tmp_path: Path) -> None:
    osekit.utils.core_utils._is_grp_supported = False
    try:
        change_owner_group(path=tmp_path, owner_group="test")
        chmod_if_needed(path=tmp_path, mode=0o775)
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.unit
def test_no_chmod_attempt_if_not_needed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    osekit.utils.core_utils._is_grp_supported = True

    monkeypatch.setattr(os, "access", lambda path, mode: mode in [os.R_OK, os.W_OK])
    monkeypatch.setattr(
        Path,
        "chmod",
        lambda path, mode: pytest.fail(
            "Call to chmod although user already has read and write permissions.",
        ),
    )

    chmod_if_needed(path=tmp_path, mode=0o775)


@pytest.mark.unit
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
def test_chmod_called_if_missing_permissions(
    new_mode: int,
    path_access: list[int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    osekit.utils.core_utils._is_grp_supported = True

    file_mode = 0o664

    def set_mode(mode: int) -> None:
        nonlocal file_mode
        file_mode = mode

    monkeypatch.setattr(os, "access", lambda path, mode: mode in [os.R_OK])
    monkeypatch.setattr(Path, "chmod", lambda path, mode: set_mode(mode))

    chmod_if_needed(path=tmp_path, mode=new_mode)
    assert file_mode == new_mode


@pytest.mark.integ
def test_error_logged_if_no_chmod_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    osekit.utils.core_utils._is_grp_supported = True
    monkeypatch.setattr(os, "access", lambda path, mode: mode in [])

    def raise_permission_error() -> None:
        raise PermissionError

    monkeypatch.setattr(Path, "chmod", lambda path, mode: raise_permission_error())

    with caplog.at_level(logging.WARNING), pytest.raises(PermissionError) as e:
        assert chmod_if_needed(path=tmp_path, mode=0o775) == e

    assert f"You do not have the permission to write to {tmp_path}" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old_group", "existing_groups", "new_group"),
    [
        pytest.param(
            "ensta",
            ["ensta", "gosmose", "other"],
            "other",
            id="change_to_new_group",
        ),
        pytest.param(
            "gosmose",
            ["ensta", "gosmose", "other"],
            "gosmose",
            id="change_to_same_group",
        ),
    ],
)
def test_change_owner_group(
    old_group: str,
    existing_groups: list[str],
    new_group: str,
    tmp_path: Path,
    patch_grp_module: MagicMock,
) -> None:
    osekit.utils.core_utils._is_grp_supported = True

    patch_grp_module.groups = existing_groups
    patch_grp_module.active_group = {"gid": existing_groups.index(old_group)}

    change_owner_group(path=tmp_path, owner_group=new_group)

    assert patch_grp_module.groups[patch_grp_module.active_group["gid"]] == new_group


@pytest.mark.integ
def test_change_owner_group_keyerror_is_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    patch_grp_module: MagicMock,
) -> None:
    osekit.utils.core_utils._is_grp_supported = True

    patch_grp_module.groups = ["ensta", "gosmose", "other"]
    patch_grp_module.active_group = {"gid": 1}

    with pytest.raises(KeyError) as e, caplog.at_level(logging.ERROR):
        assert change_owner_group(path=tmp_path, owner_group="non_existing_group") == e

    assert "Group non_existing_group does not exist." in caplog.text
    assert patch_grp_module.groups[patch_grp_module.active_group["gid"]] == "gosmose"


@pytest.mark.integ
def test_change_owner_group_permission_error_is_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    patch_grp_module: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    osekit.utils.core_utils._is_grp_supported = True

    existing_groups = ["ensta", "gosmose", "other"]
    old_group = "other"
    new_group = "gosmose"

    patch_grp_module.groups = existing_groups
    patch_grp_module.active_group = {"gid": existing_groups.index(old_group)}

    def chmod_permission_error(*args: any) -> None:
        raise PermissionError

    monkeypatch.setattr(os, "chown", chmod_permission_error, raising=False)

    with pytest.raises(PermissionError) as e, caplog.at_level(logging.ERROR):
        assert change_owner_group(path=tmp_path, owner_group=new_group) == e

    assert (
        f"You do not have the permission to change the owner of {tmp_path}."
        in caplog.text
    )
    assert patch_grp_module.groups[patch_grp_module.active_group["gid"]] == old_group
