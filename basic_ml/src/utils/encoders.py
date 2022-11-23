from json import JSONEncoder
from typing import Any

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)

        return JSONEncoder.default(self, obj)
