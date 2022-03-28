import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from config import RunConfig

import dask.dataframe as dd

logger = logging.getLogger(__name__)


def load_cached_object(
    cache_path: Path,
    run_config: RunConfig,
    filename: str,
    ext: str = "parquet",
    to_pandas: bool = False,
    chunksize=None,
) -> Any:
    target_file = Path(
        cache_path,
        str(run_config.country),
        str(run_config.product_group),
        str(run_config.period),
        f"{filename}.{ext}",
    )

    if ext not in ["parquet", "csv", "pkl"]:
        raise ValueError(f"Format '{ext}'not supported")

    if ext == "parquet":
        ddf = (
            dd.read_parquet(
                target_file.resolve(), engine="pyarrow", chunksize=chunksize
            )
            if target_file.exists()
            else None
        )
        if to_pandas:
            return ddf.compute()
        return ddf
    if ext == "pkl":
        try:
            with open(target_file.resolve(), "rb") as file_pointer:
                return pickle.load(file_pointer)
        except (OSError, EOFError) as ex:  # noqa: F841
            logger.debug(
                "Could not load pickle file '%s' from filesystem cache.",
                target_file.resolve(),
            )
            return None


def export_dataframe_to_fs(
    cache_path: Path,
    data_df: pd.DataFrame,
    run_config: RunConfig,
    filename: str,
    ext: str = "parquet",
) -> None:
    target_dir = Path(
        cache_path,
        str(run_config.country),
        str(run_config.product_group),
        str(run_config.period),
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = str(Path(target_dir, f"{filename}.{ext}").resolve())
    # with target_file.open("wb+") as file_pointer:
    if ext == "parquet":
        data_df.to_parquet(
            target_file, compression="gzip", engine="pyarrow", index=False
        )
    else:
        raise ValueError(f"Format '{ext}'not supported")


def export_object_to_fs(
    cache_path: Path, obj: object, run_config: RunConfig, filename: str
) -> None:
    target_dir = Path(
        cache_path,
        str(run_config.country),
        str(run_config.product_group),
        str(run_config.period),
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = Path(target_dir, f"{filename}.pkl").resolve()
    with target_file.open("wb+") as fp:
        pickle.dump(obj, fp)
