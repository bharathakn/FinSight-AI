"""Process-level defaults applied before any heavy import (also set in the Docker image).

ARROW_DEFAULT_MEMORY_POOL=system  pyarrow returns freed parquet buffers to the OS (~100 MB lower RSS, D-034)
OMP_NUM_THREADS=1                 one request = one core; concurrency comes from requests, not BLAS threads
"""
import os

for k, v in {"ARROW_DEFAULT_MEMORY_POOL": "system", "OMP_NUM_THREADS": "1"}.items():
    os.environ.setdefault(k, v)
