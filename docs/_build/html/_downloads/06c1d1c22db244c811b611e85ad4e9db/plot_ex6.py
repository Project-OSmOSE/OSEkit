"""
==============================================
Generate spectrograms
==============================================

This code will show you how to compute spectrograms
"""

import os
import matplotlib.pyplot as plt


#####################################################
# As this is an example, we have already worked out where
# we need to crop for the active region we want to showcase.

start_time = Time('2012-09-24T14:56:03', scale='utc', format='isot')
bottom_left = SkyCoord(-500*u.arcsec, -275*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")
top_right = SkyCoord(150*u.arcsec, 375*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")

#####################################################
# Now construct the cutout from the coordinates above
# above using the `~sunpy.net.jsoc.attrs.Cutout` attribute.

cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)


