#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:35:30 2023

@author: Siro Moreno

Here we define functions needed to operate with bottom-up pseudospectral
collocations schemes. In order to keep the best accuracy in interpolations, 
barycentric formulas are constructed.
"""

from .pseudospectral import LG, LGL, LGR
from .util import gauss_rep_integral

