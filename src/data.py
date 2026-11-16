"""Daily US equity returns from the Kenneth R. French Data Library (free, no key).

https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

The study reads the frozen copy in data/snapshot. To download it again:

    python -m src.data

Columns of the snapshot, all daily excess returns over the one month bill rate, as decimals:
  market   the value-weighted US market (the Mkt-RF factor)
  large    the tenth of stocks with the largest market value, value weighted
  small    the tenth of stocks with the smallest market value, value weighted
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "snapshot" / "daily_excess_returns.csv.gz"
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FACTORS = "F-F_Research_Data_Factors_daily"
SIZE = "Portfolios_Formed_on_ME_Daily"
SERIES = ["market", "large", "small"]


def parse_block(text: str, block: int = 0) -> pd.DataFrame:
    """Return one table of a Ken French CSV as decimals indexed by date.

    A table is a header line that starts with a comma, followed by rows that start with an
    eight digit date. Missing values are coded -99.99 or -999.
    """
    tables, header, rows = [], None, []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 1 and parts[0] == "" and not rows:
            header = parts[1:]
        elif header is not None and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
        elif rows:
            tables.append((header, rows))
            header, rows = None, []
    if rows:
        tables.append((header, rows))
    header, rows = tables[block]
    df = pd.DataFrame([r[1:len(header) + 1] for r in rows], columns=header,
                      index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"))
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.mask(df <= -99.99) / 100.0


def read_zip(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def build(factors_text: str, size_text: str) -> pd.DataFrame:
    f = parse_block(factors_text)
    s = parse_block(size_text)          # first table: value weighted returns
    out = pd.DataFrame({"market": f["Mkt-RF"], "large": s["Hi 10"] - f["RF"], "small": s["Lo 10"] - f["RF"]})
    out.index.name = "date"
    return out.dropna()


def download(name: str) -> bytes:
    req = Request(URL.format(name=name), headers={"User-Agent": "curl/8.5.0"})
    with urlopen(req, timeout=90) as r:
        return r.read()


def refresh(factors_zip: bytes | None = None, size_zip: bytes | None = None) -> Path:
    df = build(read_zip(factors_zip or download(FACTORS)), read_zip(size_zip or download(SIZE)))
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(SNAPSHOT, compression={"method": "gzip", "mtime": 0})
    return SNAPSHOT


def load() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT, index_col="date", parse_dates=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:              # two local zip files instead of a download
        print(refresh(Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]).read_bytes()))
    else:
        print(refresh())
