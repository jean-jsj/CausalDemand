from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class PackagingError(RuntimeError):
    pass


class HashingWriter:
    def __init__(self, path: Path | None):
        self.f = open(path, "wb") if path is not None else None
        self.h = hashlib.sha256()

    def write(self, s: str) -> int:
        b = s.encode("utf-8")
        self.h.update(b)
        if self.f is not None:
            self.f.write(b)
        return len(s)

    def close(self) -> str:
        if self.f is not None:
            self.f.close()
        return self.h.hexdigest()


def csv_writer(sink):
    return csv.writer(sink, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


FORMATS = {"string": str, "int": str, "bool": lambda v: "True" if v else "False", "repr": repr,
           "fixed6": "%.6f".__mod__, "sci18": "%.18e".__mod__, "float_of_int": lambda v: repr(float(v))}


def format_column(arr: pa.ChunkedArray | pa.Array, how: str) -> list[str]:
    vals = arr.to_pylist()
    if arr.null_count == 0 and how in FORMATS:
        return vals if how == "string" else list(map(FORMATS[how], vals))
    if how == "string":
        return ["" if v is None else v for v in vals]
    if how == "int":
        return ["" if v is None else str(v) for v in vals]
    if how == "bool":
        return ["" if v is None else ("True" if v else "False") for v in vals]
    if how == "repr":
        return ["" if v is None else repr(v) for v in vals]
    if how == "fixed6":
        return ["" if v is None else "%.6f" % v for v in vals]
    if how == "sci18":
        return ["" if v is None else "%.18e" % v for v in vals]
    if how == "float_of_int":
        return ["" if v is None else repr(float(v)) for v in vals]
    raise PackagingError(f"unknown csv_text {how!r}")


NEEDS_QUOTE = re.compile(r'[",\n\r]')


def write_csv_text(pq_path: Path, sink, columns: list[dict]) -> int:
    pf = pq.ParquetFile(pq_path)
    names = [c["name"] for c in columns]
    if pf.schema_arrow.names != names:
        raise PackagingError(f"{pq_path}: columns differ from the manifest")
    rows = 0
    w = csv_writer(sink)
    w.writerow(names)
    for batch in pf.iter_batches(batch_size=200_000):
        cols = [format_column(batch.column(i), c["csv_text"]) for i, c in enumerate(columns)]
        quote = any(c["csv_text"] == "string" and any(NEEDS_QUOTE.search(v) or v == "" for v in col)
                    for c, col in zip(columns, cols))
        if quote:
            w.writerows(zip(*cols))
        else:
            sink.write("".join(",".join(r) + "\n" for r in zip(*cols)))
        rows += batch.num_rows
    return rows


def parquet_to_csv(pq_path: Path, csv_path: Path, columns: list[dict]) -> int:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        return write_csv_text(pq_path, f, columns)
