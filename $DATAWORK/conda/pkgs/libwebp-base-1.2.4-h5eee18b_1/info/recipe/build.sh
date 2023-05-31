#!/bin/bash
set -ex

# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/libtool/build-aux/config.* .

# The libwebp build script doesn't pick all the other libraries up on its own
# (even though it should by using PREFIX), so pass all the necessary parameters
# for finding other imaging libraries to the configure script.
./configure \
    --disable-dependency-tracking \
    --disable-gl \
    --disable-static \
    --enable-libwebpdecoder \
    --enable-libwebpdemux \
    --enable-libwebpmux \
    --prefix=${PREFIX} \

make -j${CPU_COUNT}

if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
make check
fi

make install

rm -f $PREFIX/bin/cwebp
rm -f $PREFIX/bin/dwebp
rm -f $PREFIX/bin/gif2webp
rm -f $PREFIX/bin/img2webp
rm -f $PREFIX/bin/webpinfo
rm -f $PREFIX/bin/webpmux
