

set -ex



djpeg -dct int -ppm -outfile testout.ppm testorig.jpg
test -f ${PREFIX}/lib/pkgconfig/libjpeg.pc
exit 0
