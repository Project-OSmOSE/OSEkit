

set -ex



pip check
python -c "import contourpy as c; c.contour_generator(z=[[0, 1], [2, 3]]).lines(0.9)"
exit 0
