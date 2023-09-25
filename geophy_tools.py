# -*- coding: utf-8 -*-

"""
Tools:
reading and writing binary files
"""

import numpy as np
import math

"""
Read and write binary files (32 bits)
"""

def readbin(flnam,nz,nx):
    # Read binary file (32 bits)
    with open(flnam,"rb") as fl:
        im = np.fromfile(fl, dtype=np.float32)
    im = im.reshape(nz,nx,order='F')
    return im

def writebin(inp,flnam):
    # Write binary fila on disk (32 bits)
    with open(flnam,"wb") as fl:
        inp.T.astype('float32').tofile(fl)
