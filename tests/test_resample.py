import os
from pathlib import Path
import platform
import soundfile as sf
import shutil
from OSmOSE.cluster.resample import resample
import pytest


@pytest.mark.skipif(platform.system() == "Windows", reason="Sox is linux only")
def test_resample(input_dir: Path, output_dir: Path):
    for i in range(3):
        wav_file = input_dir.joinpath(f"test{i}.wav")
        shutil.copyfile(input_dir.joinpath("test.wav"), wav_file)

    for sr in [100, 500, 8000]:
        resample(input_dir=input_dir, output_dir=output_dir, target_sr=sr)

        # check that all resampled files exist and have the correct properties
        for i in range(3):
            output_file = output_dir.joinpath(f"test{i}.wav")
            assert output_file.is_file()
            assert sf.info(output_file).sample_rate == sr
            assert sf.info(output_file).channels == 1
            assert sf.info(output_file).frames == 900
            assert sf.info(output_file).duration == 3.0

        assert len(os.listdir(output_dir)) == 4
        # check that the original files were not modified
        for i in range(3):
            input_file = input_dir.joinpath(f"test{i}.wav")
            assert sf.info(input_file).sample_rate == 300
