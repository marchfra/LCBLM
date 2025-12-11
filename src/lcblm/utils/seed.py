import os
import random

import numpy as np
import torch

from lcblm._logging import utils_logger as logger


def set_seeds(seed: int = 42) -> None:
    """Set the seed for all sources of randomness."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002 [torch might use legacy np.random calls]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Set seed for all sources of randomness.", extra={"seed": seed})
