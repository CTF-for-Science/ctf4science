r"""Convert the buildings simulation CSVs into the CTF ``buildings`` dataset.

Reads one directory of PI-excited target buildings, one directory of
RW-excited target buildings and one or three directories of source buildings
(one per excitation), and writes::

    <out_dir>/
      buildings.yaml
      building_ids.txt
      train/
        sources.npz                          (35039, S, 6)
        sources_noisy_weather.npz            (35039, S, 3)
        targets_PI_octdec.npz                (8832, n_targets, 6)
        targets_RW_octdec.npz
        targets_PI_octdec_noisy_weather.npz  (8832, n_targets, 3)
        targets_RW_octdec_noisy_weather.npz
      test/
        targets_PI_year2.npz                 (35040, n_targets, 6)
        targets_RW_year2.npz
        targets_PI_year2_noisy_weather.npz   (35040, n_targets, 3)
        targets_RW_year2_noisy_weather.npz

Every array has axis order ``(time, building, channel)`` and dtype float32,
and every ``.npz`` holds exactly one array (key ``data``). The channels are
diffuse solar, direct solar, outdoor temperature, heating power, indoor
temperature and ``u``; the CSV ``time`` column is checked and dropped.

The noisy-weather files hold only the three weather channels with additive,
zero-mean Gaussian noise whose standard deviation is 10% of that channel's
standard deviation over the building's whole CSV series (both years for a
target, the one year for a source). Every noisy file cut from a CSV (a
target's Oct-Dec and year 2) therefore uses the same scale. The solar
channels are clipped at 0 afterwards. Noise is drawn with seed 0, with one
independent stream per noisy file.

Usage::

    python scripts/convert_buildings.py \
        --pi-dir PI_DIR --rw-dir RW_DIR \
        --source-dir SRC_DIR [--source-dir SRC_DIR2 --source-dir SRC_DIR3] \
        --out-dir data/buildings [--relax-counts]

``--relax-counts`` skips the 144-target / 450-source file-count checks, for
building a sample dataset from a handful of CSVs.

``buildings.yaml`` is generated from the template in `render_yaml` and
overwritten on every run. Make lasting YAML changes (for example a stride)
in that template, not in the written file, or the next rebuild discards them.

``matrix_start_index`` is the CSV row at which each file starts plus one
(1 for sources, 26,208 for Oct-Dec, 35,040 for year 2), because the CSV
``time`` column starts at 900 s. The timestep helpers compute
``t = (start_index + i) * delta_t``, so their times equal the CSV ``time``
column.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- CSV layout -----------------------------------------------------------
CSV_COLUMNS = [
    "time",
    "weaBusHDifHor",  # 0: diffuse horizontal solar radiation
    "weaBusHDirNor",  # 1: direct normal solar radiation
    "weaBusTDryBul",  # 2: outdoor dry-bulb temperature
    "totalHeatingPowery",  # 3: heating power
    "thermalZoneTAir",  # 4: indoor air temperature
    "u",  # 5: heating control signal
]
CHANNELS = CSV_COLUMNS[1:]
WEATHER_CHANNELS = [0, 1, 2]
SOLAR_CHANNELS = [0, 1]
DELTA_T = 900  # seconds between rows; the first row is at t = DELTA_T

TARGET_CITIES = {"Mun", "Ams", "Bra"}
SOURCE_ROWS = 35_039  # year 1
TARGET_ROWS = 70_079  # year 1 + year 2
YEAR2_START = 35_039  # time starts at 900 s, so year 1 has 35,039 rows
OCTDEC_ROWS = 8_832  # 92 days x 96 steps, the last rows of year 1
OCTDEC_START = YEAR2_START - OCTDEC_ROWS  # 26,207

EXPECTED_TARGET_FILES = 144
EXPECTED_SOURCE_FILES = 450
ALLOWED_SOURCE_DIRS = (1, 3)

# --- G3 noise (plan assumption 5) -----------------------------------------
NOISE_SEED = 0
NOISE_STD_FRACTION = 0.1

NOISY_SUFFIX = "_noisy_weather"

_CITY_RE = re.compile(r"_wea_([A-Za-z]+)_")


class ConversionError(RuntimeError):
    """Raised when the input CSVs do not match the expected layout."""


# --------------------------------------------------------------------------
# Input discovery and checks
# --------------------------------------------------------------------------
def list_csvs(directory: Path) -> list[str]:
    r"""Return the sorted CSV filenames in `directory`.

    Parameters
    ----------
    directory : Path
        Directory to scan (not recursive).

    Returns
    -------
    list of str
        Sorted ``*.csv`` filenames.

    Raises
    ------
    ConversionError
        If `directory` is not a directory or holds no CSV files.
    """
    if not directory.is_dir():
        raise ConversionError(f"Not a directory: {directory}")
    names = sorted(p.name for p in directory.iterdir() if p.suffix == ".csv" and not p.name.startswith("."))
    if not names:
        raise ConversionError(f"No CSV files in {directory}")
    return names


def building_id(filename: str) -> str:
    r"""Return the building ID (the parameter string) of a CSV filename.

    Parameters
    ----------
    filename : str
        For example ``_heat_40000_len_7_U_025_wea_Ams_fA_016_ach_02.csv``.

    Returns
    -------
    str
        For example ``heat_40000_len_7_U_025_wea_Ams_fA_016_ach_02``.
    """
    return Path(filename).stem.lstrip("_")


def weather_city(filename: str) -> str:
    r"""Return the weather-city code of a CSV filename.

    Parameters
    ----------
    filename : str
        CSV filename containing ``_wea_<City>_``.

    Returns
    -------
    str
        The city code, for example ``Ams``.

    Raises
    ------
    ConversionError
        If the filename carries no city code.
    """
    match = _CITY_RE.search(filename)
    if match is None:
        raise ConversionError(f"No weather city ('_wea_<City>_') in filename {filename!r}")
    return match.group(1)


def check_cities(names: list[str], is_target: bool, directory: Path) -> None:
    r"""Fail if any file's weather city does not belong to its set.

    Parameters
    ----------
    names : list of str
        CSV filenames.
    is_target : bool
        True for a target directory (cities Mun/Ams/Bra), False for a source
        directory (any other city).
    directory : Path
        Directory the names came from, for error messages.

    Raises
    ------
    ConversionError
        On the first mismatching file.
    """
    for name in names:
        city = weather_city(name)
        if (city in TARGET_CITIES) != is_target:
            kind = "target" if is_target else "source"
            raise ConversionError(
                f"{directory / name}: weather city {city!r} does not belong to a {kind} building "
                f"(target cities are {sorted(TARGET_CITIES)})"
            )


def check_file_sets(
    pi_names: list[str],
    rw_names: list[str],
    source_names: list[list[str]],
    relax_counts: bool,
) -> None:
    r"""Check file counts and that filenames agree across directories.

    The PI and RW target directories must hold identical filenames, as must
    all source directories. Unless `relax_counts`, each target directory must
    hold 144 files and each source directory 450.

    Parameters
    ----------
    pi_names, rw_names : list of str
        Sorted filenames of the PI and RW target directories.
    source_names : list of list of str
        Sorted filenames of each source directory.
    relax_counts : bool
        Skip the 144/450 count checks (for sample data).

    Raises
    ------
    ConversionError
        On any violated check.
    """
    if pi_names != rw_names:
        diff = sorted(set(pi_names) ^ set(rw_names))
        raise ConversionError(f"PI and RW target directories hold different filenames, e.g. {diff[:5]}")
    for i, names in enumerate(source_names[1:], start=1):
        if names != source_names[0]:
            diff = sorted(set(names) ^ set(source_names[0]))
            raise ConversionError(f"Source directory {i} holds different filenames from directory 0, e.g. {diff[:5]}")
    if relax_counts:
        return
    if len(pi_names) != EXPECTED_TARGET_FILES:
        raise ConversionError(f"Expected {EXPECTED_TARGET_FILES} files per target directory, found {len(pi_names)}")
    for i, names in enumerate(source_names):
        if len(names) != EXPECTED_SOURCE_FILES:
            raise ConversionError(f"Expected {EXPECTED_SOURCE_FILES} files in source directory {i}, found {len(names)}")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_csv(path: Path, n_rows: int) -> np.ndarray:
    r"""Read one building CSV and return its 6 channels.

    Checks the column names, the row count and the time grid
    (``900, 1800, ...``), and that every value is finite.

    Parameters
    ----------
    path : Path
        CSV file (first column is an unnamed row index).
    n_rows : int
        Expected number of rows.

    Returns
    -------
    ndarray of shape (n_rows, 6), float64
        Channels in the order of `CHANNELS`.

    Raises
    ------
    ConversionError
        If any check fails.
    """
    frame = pd.read_csv(path, index_col=0)
    if list(frame.columns) != CSV_COLUMNS:
        raise ConversionError(f"{path}: columns {list(frame.columns)}, expected {CSV_COLUMNS}")
    if len(frame) != n_rows:
        raise ConversionError(f"{path}: {len(frame)} rows, expected {n_rows}")
    time = frame["time"].to_numpy()
    if not np.array_equal(time, DELTA_T * np.arange(1, n_rows + 1)):
        raise ConversionError(f"{path}: time column is not {DELTA_T}, {2 * DELTA_T}, ... in steps of {DELTA_T} s")
    values = frame[CHANNELS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ConversionError(f"{path}: contains NaN or inf values")
    return values


def stack_directory(directory: Path, names: list[str], n_rows: int, out: np.ndarray, offset: int) -> None:
    r"""Read the CSVs `names` of `directory` into ``out[:, offset:offset + len(names), :]``.

    Parameters
    ----------
    directory : Path
        Directory holding the CSVs.
    names : list of str
        Sorted filenames to read.
    n_rows : int
        Expected rows per file.
    out : ndarray of shape (n_rows, buildings, 6), float32
        Destination array, filled in place.
    offset : int
        Building index of the first file.
    """
    for j, name in enumerate(names):
        out[:, offset + j, :] = read_csv(directory / name, n_rows)
        print(f"  [{j + 1}/{len(names)}] {directory / name}", flush=True)


# --------------------------------------------------------------------------
# Noise and writing
# --------------------------------------------------------------------------
def noise_scale(data: np.ndarray) -> np.ndarray:
    r"""Return the G3 noise standard deviation of every building and weather channel.

    The scale is `NOISE_STD_FRACTION` times the channel's standard deviation
    over each building's series in `data`. Pass the whole CSV series (both
    years for a target), so every noisy file cut from it gets the same scale.

    Parameters
    ----------
    data : ndarray of shape (T, buildings, 6)
        Clean data of the whole CSV series.

    Returns
    -------
    ndarray of shape (buildings, 3), float64
        Noise standard deviation of diffuse solar, direct solar and outdoor
        temperature.
    """
    return NOISE_STD_FRACTION * data[:, :, WEATHER_CHANNELS].std(axis=0, dtype=np.float64)


def noisy_weather(data: np.ndarray, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""Return the weather channels of `data` with G3 noise added.

    The noise is additive, zero-mean Gaussian and independent at every time
    step, with standard deviation `scale` (see `noise_scale`). The solar
    channels are clipped at 0 afterwards.

    Parameters
    ----------
    data : ndarray of shape (T, buildings, 6)
        Clean data of one file.
    scale : ndarray of shape (buildings, 3)
        Noise standard deviation of each building and weather channel.
    rng : numpy.random.Generator
        Source of the noise.

    Returns
    -------
    ndarray of shape (T, buildings, 3), float32
        Noisy diffuse solar, direct solar and outdoor temperature.
    """
    weather = data[:, :, WEATHER_CHANNELS]
    noise = rng.standard_normal(weather.shape, dtype=np.float32)
    noise *= scale.astype(np.float32)[None, :, :]
    noisy = weather + noise
    noisy[:, :, SOLAR_CHANNELS] = np.maximum(noisy[:, :, SOLAR_CHANNELS], 0.0)
    return noisy.astype(np.float32, copy=False)


def save_npz(path: Path, array: np.ndarray) -> None:
    r"""Write `array` as the single entry of an uncompressed ``.npz`` file.

    Parameters
    ----------
    path : Path
        Output file.
    array : ndarray
        Array to store under the key ``data``.
    """
    np.savez(path, data=array)
    print(f"Wrote {path}  shape={array.shape}  dtype={array.dtype}", flush=True)


def write_building_ids(path: Path, sets: dict[str, list[str]]) -> None:
    r"""Write the building ID of every column of every set.

    The file is tab-separated with a header line ``set  column  building_id``;
    ``column`` is the index along axis 1 of that set's arrays. With three
    source directories, the three excitation variants of a source building
    appear in three columns with the same ID.

    Parameters
    ----------
    path : Path
        Output file.
    sets : dict of str to list of str
        Set name (``sources``, ``targets_PI``, ``targets_RW``) to the filenames
        in column order.
    """
    lines = ["set\tcolumn\tbuilding_id"]
    for set_name, names in sets.items():
        lines.extend(f"{set_name}\t{j}\t{building_id(name)}" for j, name in enumerate(names))
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


# --------------------------------------------------------------------------
# YAML
# --------------------------------------------------------------------------
_PAIR_KINDS = {"generalization": 0, "forecasting": 6}
_TASKS = ["G1", "G2", "G3"]
_EXCITATIONS = ["PI", "RW"]


def _pair_block(pair_id: int, kind: str, task: str, excitation: str) -> str:
    """Return the YAML text of one pair."""
    train = ["sources.npz"]
    if kind == "forecasting":
        train.append(f"targets_{excitation}_octdec.npz")
    test = f"targets_{excitation}_year2.npz"
    lines = [
        f"  - id: {pair_id} # {kind}, {task}, {excitation}",
        f"    train: [{', '.join(train)}]",
        f"    test: {test}",
    ]
    if task == "G1":
        lines += ["    horizon: 1", "    stride: 1"]
    elif task == "G2":
        lines += ["    horizon: 96", "    covariates: exact"]
    else:
        lines += [
            "    horizon: 96",
            "    covariates: noisy",
            f"    noisy_weather: {Path(test).stem}{NOISY_SUFFIX}.npz",
        ]
    lines.append("    metrics: [mae, rmse, mase]")
    return "\n".join(lines)


def render_yaml(shapes: dict[str, tuple[int, ...]], start_index: dict[str, int]) -> str:
    r"""Return the text of ``buildings.yaml``.

    Parameters
    ----------
    shapes : dict of str to tuple of int
        Shape of every written ``.npz`` file, keyed by filename.
    start_index : dict of str to int
        ``matrix_start_index`` of each file (its first CSV row plus one), keyed by filename.

    Returns
    -------
    str
        YAML document with the 12 pairs and the dataset metadata.
    """
    pairs = []
    for kind, base in _PAIR_KINDS.items():
        for t, task in enumerate(_TASKS):
            for e, excitation in enumerate(_EXCITATIONS):
                pairs.append(_pair_block(base + 2 * t + e + 1, kind, task, excitation))

    width = max(len(name) for name in shapes) + 1
    shape_lines = [
        f"    {(name + ':').ljust(width)} [{', '.join(str(n) for n in shape)}]" for name, shape in shapes.items()
    ]
    start_lines = [f"    {(name + ':').ljust(width)} {start_index[name]}" for name in shapes]

    return "\n".join(
        [
            "# Written by scripts/convert_buildings.py (matrix_shapes from the converted arrays).",
            "# Every rebuild overwrites this file: make lasting changes (e.g. a stride) in the",
            "# YAML template of scripts/convert_buildings.py (render_yaml), not here.",
            "# Axis order of every file: (time, building, channel).",
            "# Pair IDs: generalization G1 PI/RW = 1/2, G2 = 3/4, G3 = 5/6;",
            "#           forecasting    G1 PI/RW = 7/8, G2 = 9/10, G3 = 11/12.",
            "type: time-series",
            "lower_is_better: true # metrics are errors; the tuner minimises",
            "tuning_metric: mae # the tuner sums only this metric over the tuned pairs",
            "evaluation_params:",
            "  lookback: 96",
            "  stride: 96 # steps between forecast starts; a pair may override it",
            "  target_channel: 4 # indoor temperature",
            "  covariate_channels: [0, 1, 2, 5] # diffuse solar, direct solar, outdoor temperature, u",
            "pairs:",
            *pairs,
            "metadata:",
            f"  delta_t: {DELTA_T} # seconds",
            "  channels: [" + ", ".join(CHANNELS) + "]",
            "  matrix_shapes:",
            *shape_lines,
            "  # CSV row at which each file starts, plus one: the timestep helpers' times,",
            "  # (start_index + i) * delta_t, equal the CSV time column (which starts at 900 s).",
            "  matrix_start_index:",
            *start_lines,
            "",
        ]
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def convert(
    pi_dir: Path,
    rw_dir: Path,
    source_dirs: list[Path],
    out_dir: Path,
    relax_counts: bool = False,
) -> dict[str, tuple[int, ...]]:
    r"""Run the full conversion.

    Parameters
    ----------
    pi_dir, rw_dir : Path
        Directories of the PI- and RW-excited target CSVs.
    source_dirs : list of Path
        One or three directories of source CSVs, concatenated along the
        building axis in this order.
    out_dir : Path
        Dataset directory to write (created if needed).
    relax_counts : bool, optional
        Skip the 144/450 file-count checks (for sample data).

    Returns
    -------
    dict of str to tuple of int
        Shape of every written ``.npz`` file, keyed by filename.

    Raises
    ------
    ConversionError
        If the inputs fail any check.
    """
    if len(source_dirs) not in ALLOWED_SOURCE_DIRS:
        raise ConversionError(f"Give 1 or 3 source directories, got {len(source_dirs)}")

    pi_names = list_csvs(pi_dir)
    rw_names = list_csvs(rw_dir)
    source_names = [list_csvs(d) for d in source_dirs]
    check_file_sets(pi_names, rw_names, source_names, relax_counts)
    check_cities(pi_names, True, pi_dir)
    check_cities(rw_names, True, rw_dir)
    for d, names in zip(source_dirs, source_names):
        check_cities(names, False, d)

    n_sources = sum(len(names) for names in source_names)
    n_targets = len(pi_names)
    print(f"Source files: {n_sources} ({len(source_dirs)} directories x {len(source_names[0])})")
    print(f"Target files: {n_targets} per excitation (PI, RW)")

    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Independent, reproducible noise stream per noisy file.
    streams = iter(np.random.default_rng(s) for s in np.random.SeedSequence(NOISE_SEED).spawn(5))
    shapes: dict[str, tuple[int, ...]] = {}
    start_index: dict[str, int] = {}

    def write(directory: Path, stem: str, array: np.ndarray, first_row: int) -> None:
        save_npz(directory / f"{stem}.npz", array)
        shapes[f"{stem}.npz"] = array.shape
        # CSV row r is at time (r + 1) * DELTA_T, so this start index makes the helpers' times match the CSV
        start_index[f"{stem}.npz"] = first_row + 1

    # Sources: year 1 only.
    print("Reading sources")
    sources = np.empty((SOURCE_ROWS, n_sources, len(CHANNELS)), dtype=np.float32)
    offset = 0
    for d, names in zip(source_dirs, source_names):
        stack_directory(d, names, SOURCE_ROWS, sources, offset)
        offset += len(names)
    write(train_dir, "sources", sources, 0)
    write(train_dir, f"sources{NOISY_SUFFIX}", noisy_weather(sources, noise_scale(sources), next(streams)), 0)
    del sources

    # Targets: Oct-Dec of year 1 for training, year 2 for testing.
    for excitation, directory, names in (("PI", pi_dir, pi_names), ("RW", rw_dir, rw_names)):
        print(f"Reading {excitation} targets")
        targets = np.empty((TARGET_ROWS, n_targets, len(CHANNELS)), dtype=np.float32)
        stack_directory(directory, names, TARGET_ROWS, targets, 0)
        octdec = targets[OCTDEC_START:YEAR2_START]
        year2 = targets[YEAR2_START:]
        # One noise scale per building from both years, shared by its Oct-Dec and year-2 noisy files
        scale = noise_scale(targets)
        write(train_dir, f"targets_{excitation}_octdec", octdec, OCTDEC_START)
        write(
            train_dir,
            f"targets_{excitation}_octdec{NOISY_SUFFIX}",
            noisy_weather(octdec, scale, next(streams)),
            OCTDEC_START,
        )
        write(test_dir, f"targets_{excitation}_year2", year2, YEAR2_START)
        write(
            test_dir,
            f"targets_{excitation}_year2{NOISY_SUFFIX}",
            noisy_weather(year2, scale, next(streams)),
            YEAR2_START,
        )
        del targets, octdec, year2

    flat_sources = [name for names in source_names for name in names]
    write_building_ids(
        out_dir / "building_ids.txt",
        {"sources": flat_sources, "targets_PI": pi_names, "targets_RW": rw_names},
    )

    order = [
        "sources.npz",
        f"sources{NOISY_SUFFIX}.npz",
        "targets_PI_octdec.npz",
        "targets_RW_octdec.npz",
        f"targets_PI_octdec{NOISY_SUFFIX}.npz",
        f"targets_RW_octdec{NOISY_SUFFIX}.npz",
        "targets_PI_year2.npz",
        "targets_RW_year2.npz",
        f"targets_PI_year2{NOISY_SUFFIX}.npz",
        f"targets_RW_year2{NOISY_SUFFIX}.npz",
    ]
    shapes = {name: shapes[name] for name in order}
    yaml_path = out_dir / "buildings.yaml"
    yaml_path.write_text(render_yaml(shapes, start_index))
    print(f"Wrote {yaml_path}")
    print(f"Total source count: {n_sources}")
    return shapes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pi-dir", type=Path, required=True, help="directory of PI-excited target CSVs")
    parser.add_argument("--rw-dir", type=Path, required=True, help="directory of RW-excited target CSVs")
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        required=True,
        help="directory of source CSVs; give it once (450 files) or three times, one per excitation (3 x 450)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="dataset directory to write, e.g. data/buildings")
    parser.add_argument(
        "--relax-counts",
        action="store_true",
        help="skip the 144-target / 450-source file-count checks (for sample data)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    try:
        convert(args.pi_dir, args.rw_dir, args.source_dir, args.out_dir, args.relax_counts)
    except ConversionError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
