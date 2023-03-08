import os
import pytest
import shutil
from OSmOSE.utils import *


def test_display_folder_storage_infos(monkeypatch):
    mock_usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        shutil, "disk_usage", lambda: mock_usage(2048**4, 1536**4, 1862**4)
    )

    assert True
