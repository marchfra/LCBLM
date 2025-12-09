import gc

import torch

from lcblm._logging import utils_logger as logger


def log_gpu_memory(
    allocated: float,
    reserved: float,
    msg: str = "GPU memory",
) -> None:
    """Log allocated and reserved GPU memory."""
    logger.info(
        "GPU Memory [GiB]",
        extra={"msg": msg, "allocated": allocated, "reserved": reserved},
    )


def free_gpu_memory(objects: list[str] | None = None) -> None:
    """Free GPU memory by deleting object references and emptying the cache."""
    if objects is not None:
        for obj in objects:
            if obj in globals():
                del globals()[obj]

    gc.collect()

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. No GPU memory to free.")
        return

    before_allocated = torch.cuda.memory_allocated() / 1024**3
    before_reserved = torch.cuda.memory_reserved() / 1024**3
    if before_allocated == 0 and before_reserved == 0:
        logger.info("GPU memory is already free.")
        return
    log_gpu_memory(before_allocated, before_reserved, "Before cleanup: ")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    after_allocated = torch.cuda.memory_allocated() / 1024**3
    after_reserved = torch.cuda.memory_reserved() / 1024**3
    log_gpu_memory(after_allocated, after_reserved, "After cleanup:  ")
    log_gpu_memory(
        after_allocated - before_allocated,
        after_reserved - before_reserved,
        "Freed GPU memory: ",
    )
