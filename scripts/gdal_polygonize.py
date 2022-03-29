#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ******************************************************************************
#  $Id$
#
#  Project:  GDAL Python Interface
#  Purpose:  Application for converting raster data to a vector polygon layer.
#  Author:   Frank Warmerdam, warmerdam@pobox.com
#
# ******************************************************************************
#  Copyright (c) 2008, Frank Warmerdam
#  Copyright (c) 2009-2013, Even Rouault <even dot rouault at spatialys.com>
#  Copyright (c) 2021, Idan Miara <idan@miara.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
# ******************************************************************************

import sys
import textwrap
from typing import Optional, Union

from osgeo import gdal
from osgeo import ogr

from osgeo_utils.auxiliary.util import GetOutputDriverFor


def gdal_polygonize(src_filename: Optional[str] = None, band_number: Union[int, str] = 1,
                    dst_filename: Optional[str] = None, driver_name: Optional[str] = None,
                    dst_layername: Optional[str] = None, dst_fieldname: Optional[str] = None,
                    quiet: bool = False, mask: str = 'default', options: Optional[list] = None,
                    connectedness8: bool = False):

    if isinstance(band_number, str) and not band_number.startswith('mask'):
        band_number = int(band_number)

    options = options or []

    if connectedness8:
        options.append('8CONNECTED=8')

    if driver_name is None:
        driver_name = GetOutputDriverFor(dst_filename, is_raster=False)

    if dst_layername is None:
        dst_layername = 'out'

    # =============================================================================
    # Open source file
    # =============================================================================

    src_ds = gdal.Open(src_filename)

    if src_ds is None:
        print('Unable to open %s' % src_filename)
        return 1

    if band_number == 'mask':
        srcband = src_ds.GetRasterBand(1).GetMaskBand()
        # Workaround the fact that most source bands have no dataset attached
        options.append('DATASET_FOR_GEOREF=' + src_filename)
    elif isinstance(band_number, str) and band_number.startswith('mask,'):
        srcband = src_ds.GetRasterBand(int(band_number[len('mask,'):])).GetMaskBand()
        # Workaround the fact that most source bands have no dataset attached
        options.append('DATASET_FOR_GEOREF=' + src_filename)
    else:
        srcband = src_ds.GetRasterBand(band_number)

    if mask == 'default':
        maskband = srcband.GetMaskBand()
    elif mask == 'none':
        maskband = None
    else:
        mask_ds = gdal.Open(mask)
        maskband = mask_ds.GetRasterBand(1)

    # =============================================================================
    #       Try opening the destination file as an existing file.
    # =============================================================================

    try:
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        dst_ds = ogr.Open(dst_filename, update=1)
        gdal.PopErrorHandler()
    except:
        dst_ds = None

    # =============================================================================
    # 	Create output file.
    # =============================================================================
    if dst_ds is None:
        drv = ogr.GetDriverByName(driver_name)
        if not quiet:
            print('Creating output %s of format %s.' % (dst_filename, driver_name))
        dst_ds = drv.CreateDataSource(dst_filename)

    # =============================================================================
    #       Find or create destination layer.
    # =============================================================================
    try:
        dst_layer = dst_ds.GetLayerByName(dst_layername)
    except:
        dst_layer = None

    dst_field: int = -1
    if dst_layer is None:

        srs = src_ds.GetSpatialRef()
        dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)

        if dst_fieldname is None:
            dst_fieldname = 'DN'

        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = 0
    else:
        if dst_fieldname is not None:
            dst_field = dst_layer.GetLayerDefn().GetFieldIndex(dst_fieldname)
            if dst_field < 0:
                print("Warning: cannot find field '%s' in layer '%s'" % (dst_fieldname, dst_layername))

    # =============================================================================
    # Invoke algorithm.
    # =============================================================================

    if quiet:
        prog_func = None
    else:
        prog_func = gdal.TermProgress_nocb

    result = gdal.Polygonize(srcband, maskband, dst_layer, dst_field, options,
                             callback=prog_func)

    srcband = None
    src_ds = None
    dst_ds = None
    mask_ds = None

    return result