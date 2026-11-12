"""Build the HIKARI-2021 corpus cache from the downloaded flow table.

Run with `python -I` (isolated mode): the input is a downloaded file, so no
module search path is taken from the environment or the working directory.
usage: python -I code/tools/build_hikari.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data_hikari as dh  # noqa: E402

c = dh.load_corpus(verbose=True)
print({k: getattr(v, "shape", None) for k, v in c.items()})
