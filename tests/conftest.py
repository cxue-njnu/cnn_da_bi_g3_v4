# -*- coding: utf-8 -*-
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "source_adapted")
for p in (SRC,):
    if p not in sys.path:
        sys.path.insert(0, p)
