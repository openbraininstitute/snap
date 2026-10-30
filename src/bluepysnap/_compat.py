"""Compatibility helpers for the supported versions of the dependencies.

Keep all the version-specific code here, so it can be removed when old versions are dropped.
"""

from typing import Any

import pandas as pd

PANDAS_GE_3 = int(pd.__version__.split(".")[0]) >= 3

# Keyword arguments for pd.concat to avoid copying the data.
# The copy keyword is deprecated since pandas 3, where Copy-on-Write is always active.
CONCAT_NO_COPY: dict[str, Any] = {} if PANDAS_GE_3 else {"copy": False}
