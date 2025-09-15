<br>
<br>
<div align="center">

[![OSEkit logo](https://raw.githubusercontent.com/Project-OSmOSE/OSEkit/refs/heads/main/docs/logo/osekit_small.png)](https://github.com/Project-OSmOSE/OSEkit)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![OSmOSE logo](https://raw.githubusercontent.com/Project-OSmOSE/OSEkit/refs/heads/main/docs/logo/osmose_texte_sombre_small.png)](https://osmose.ifremer.fr/)

<br>
<br>

![version](https://img.shields.io/badge/package_version-0.3.0-orange)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)
![license](https://img.shields.io/github/license/mashape/apistatus.svg)
![Open Source Love](https://img.shields.io/badge/open%20source-♡-lightgrey)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

**OSEkit** is an open source python package dedicated to the management and analysis of data in underwater passive acoustics.

[Presentation](#presentation) •
[Getting into it](#getting-into-it) •
[Acknowledgements](#acknowledgements)
# ㅤ

</div>

### Presentation

**OSEkit** is a Python open source package designed for effortless manipulation of audio data.

It is primarily designed for working on underwater passive acoustics data, but can be used on any audio data.

**OSEkit** provides:

- Seamless, timestamp-based access across multiple files (e.g., pick a 3-minutes long segment spannign over two 1.5-minutes long files without manual file picking)
- Preprocessing utilities: resampling, normalization...
- Spectral analysis routines: compute power spectra, spectrograms, LTAS...

**OSEkit** treats your audio dataset as a continuous timeline, whatever the raw files configuration.
You request audio by time intervals and the package fetches from the raw files behind the scenes: no manual I/O juggling required.

<br>

### Getting into it

All details to start using our toolkit and make the most out of it are given in our [documentation](https://project-osmose.github.io/OSEkit/).

<br>


<sub>© OSmOSE team, 2023-present</sub>
