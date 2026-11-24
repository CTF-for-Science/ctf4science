import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

# scripts/ is not a package, so load the conversion script by path
_SCRIPT = Path(__file__).parent.parent / "scripts" / "convert_buildings.py"
_spec = importlib.util.spec_from_file_location("convert_buildings", _SCRIPT)
assert _spec is not None and _spec.loader is not None
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

# Small stand-ins for the real row and file counts (35,039 / 70,079 rows, 144 / 450 files)
SOURCE_ROWS = 11
TARGET_ROWS = 23  # year 1 (11 rows) + year 2 (12 rows)
OCTDEC_ROWS = 4
ROW_CONSTANTS = {
    "SOURCE_ROWS": SOURCE_ROWS,
    "TARGET_ROWS": TARGET_ROWS,
    "YEAR2_START": SOURCE_ROWS,
    "OCTDEC_ROWS": OCTDEC_ROWS,
    "OCTDEC_START": SOURCE_ROWS - OCTDEC_ROWS,
    "EXPECTED_TARGET_FILES": 2,
    "EXPECTED_SOURCE_FILES": 2,
}

TARGET_NAMES = ["_heat_1_len_7_U_025_wea_Ams_fA_016_ach_02.csv", "_heat_2_len_7_U_025_wea_Mun_fA_016_ach_02.csv"]
SOURCE_NAMES = ["_heat_3_len_7_U_025_wea_Ber_fA_016_ach_02.csv", "_heat_4_len_7_U_025_wea_Rom_fA_016_ach_02.csv"]


def _series(n_rows: int, seed: int) -> np.ndarray:
    """A random (n_rows, 6) building series; the solar channels are 0 at every other row, so clipping matters."""
    values = np.random.default_rng(seed).uniform(1.0, 10.0, (n_rows, 6))
    values[::2, cb.SOLAR_CHANNELS] = 0.0
    return values


def _write_dir(directory: Path, names: list[str], n_rows: int, seed: int) -> list[np.ndarray]:
    """Write one CSV per name in the authors' layout and return the written channel arrays."""
    directory.mkdir(parents=True)
    arrays = []
    for k, name in enumerate(names):
        values = _series(n_rows, seed + k)
        frame = pd.DataFrame(values, columns=cb.CHANNELS)
        frame.insert(0, "time", cb.DELTA_T * np.arange(1, n_rows + 1))
        frame.to_csv(directory / name)
        arrays.append(values)
    return arrays


def _convert(pi_dir, rw_dir, source_dirs, out_dir, relax_counts=False):
    """Run the conversion quietly."""
    with contextlib.redirect_stdout(io.StringIO()):
        return cb.convert(pi_dir, rw_dir, source_dirs, out_dir, relax_counts)


def _load(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        assert list(npz.keys()) == ["data"]
        return npz["data"]


class TestConvertBuildings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.patcher = patch.multiple(cb, **ROW_CONSTANTS)
        cls.patcher.start()

        # PI and RW hold identical data, so any difference in their noisy files comes from the noise streams
        cls.pi = _write_dir(cls.root / "PI", TARGET_NAMES, TARGET_ROWS, seed=100)
        _write_dir(cls.root / "RW", TARGET_NAMES, TARGET_ROWS, seed=100)
        # Three source directories (one per excitation) with the same filenames
        cls.sources = []
        for e in range(3):
            cls.sources += _write_dir(cls.root / f"src{e}", SOURCE_NAMES, SOURCE_ROWS, seed=10 * e)
        cls.source_dirs = [cls.root / f"src{e}" for e in range(3)]

        cls.out = cls.root / "out"
        cls.out2 = cls.root / "out2"
        for out in (cls.out, cls.out2):
            _convert(cls.root / "PI", cls.root / "RW", cls.source_dirs, out)
        with open(cls.out / "buildings.yaml") as f:
            cls.config = yaml.safe_load(f)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        cls.temp_dir.cleanup()

    def _bad_input(self, name: str, pi_names, rw_names, source_names) -> tuple[Path, Path, list[Path]]:
        """Write a fresh set of input directories under `name` with the given filenames."""
        base = self.root / name
        _write_dir(base / "PI", pi_names, TARGET_ROWS, seed=0)
        _write_dir(base / "RW", rw_names, TARGET_ROWS, seed=0)
        source_dirs = []
        for e, names in enumerate(source_names):
            _write_dir(base / f"src{e}", names, SOURCE_ROWS, seed=0)
            source_dirs.append(base / f"src{e}")
        return base / "PI", base / "RW", source_dirs

    def test_row_slicing(self):
        """Sources are year 1; Oct-Dec is the last OCTDEC_ROWS rows of year 1; year 2 is the rest."""
        np.testing.assert_array_equal(
            _load(self.out / "train" / "sources.npz"), np.stack(self.sources, axis=1).astype(np.float32)
        )
        targets = np.stack(self.pi, axis=1).astype(np.float32)
        octdec = _load(self.out / "train" / "targets_PI_octdec.npz")
        year2 = _load(self.out / "test" / "targets_PI_year2.npz")
        np.testing.assert_array_equal(octdec, targets[SOURCE_ROWS - OCTDEC_ROWS : SOURCE_ROWS])
        np.testing.assert_array_equal(year2, targets[SOURCE_ROWS:])
        self.assertEqual(year2.dtype, np.float32)
        self.assertEqual(year2.shape, (TARGET_ROWS - SOURCE_ROWS, 2, 6))

    def test_yaml_shapes_and_start_indices(self):
        """The YAML lists every file's shape, and start indices make the helper times equal the CSV time."""
        metadata = self.config["metadata"]
        self.assertEqual(len(self.config["pairs"]), 12)
        self.assertEqual(metadata["matrix_shapes"]["sources.npz"], [SOURCE_ROWS, 6, 6])
        self.assertEqual(metadata["matrix_shapes"]["targets_RW_year2_noisy_weather.npz"], [12, 2, 3])
        # CSV row r is at time (r + 1) * delta_t, so each start index is the file's first CSV row plus one
        expected = {"sources": 1, "octdec": SOURCE_ROWS - OCTDEC_ROWS + 1, "year2": SOURCE_ROWS + 1}
        self.assertEqual(len(metadata["matrix_start_index"]), 10)
        for name, start in metadata["matrix_start_index"].items():
            key = next(k for k in expected if k in name)
            self.assertEqual(start, expected[key], name)

    def test_building_ids(self):
        """building_ids.txt lists every column; the three excitation variants of a source share its ID."""
        lines = (self.out / "building_ids.txt").read_text().splitlines()
        ids = [stem.lstrip("_").removesuffix(".csv") for stem in SOURCE_NAMES]
        targets = [stem.lstrip("_").removesuffix(".csv") for stem in TARGET_NAMES]
        expected = ["set\tcolumn\tbuilding_id"]
        expected += [f"sources\t{c}\t{ids[c % 2]}" for c in range(6)]
        expected += [f"targets_PI\t{c}\t{targets[c]}" for c in range(2)]
        expected += [f"targets_RW\t{c}\t{targets[c]}" for c in range(2)]
        self.assertListEqual(lines, expected)

    def test_noise(self):
        """Noise is deterministic, keeps the solar channels >= 0 and differs between PI and RW."""
        for directory, stem in (
            ("train", "sources"),
            ("train", "targets_PI_octdec"),
            ("train", "targets_RW_octdec"),
            ("test", "targets_PI_year2"),
            ("test", "targets_RW_year2"),
        ):
            with self.subTest(file=stem):
                noisy = _load(self.out / directory / f"{stem}_noisy_weather.npz")
                clean = _load(self.out / directory / f"{stem}.npz")
                np.testing.assert_array_equal(noisy, _load(self.out2 / directory / f"{stem}_noisy_weather.npz"))
                self.assertEqual(noisy.shape, (*clean.shape[:2], 3))
                self.assertGreaterEqual(noisy[..., cb.SOLAR_CHANNELS].min(), 0.0)
                self.assertFalse(np.array_equal(noisy, clean[..., cb.WEATHER_CHANNELS]))
        pi = _load(self.out / "test" / "targets_PI_year2_noisy_weather.npz")
        rw = _load(self.out / "test" / "targets_RW_year2_noisy_weather.npz")
        np.testing.assert_array_equal(
            _load(self.out / "test" / "targets_PI_year2.npz"), _load(self.out / "test" / "targets_RW_year2.npz")
        )
        self.assertFalse(np.array_equal(pi, rw))

    def test_noise_scale_per_building(self):
        """Each noisy file uses 10% of the std over the building's whole CSV series, so Oct-Dec and year 2 share it."""
        targets = np.stack(self.pi, axis=1).astype(np.float32)
        sources = np.stack(self.sources, axis=1).astype(np.float32)
        target_scale = cb.NOISE_STD_FRACTION * targets[..., cb.WEATHER_CHANNELS].std(axis=0, dtype=np.float64)
        source_scale = cb.NOISE_STD_FRACTION * sources[..., cb.WEATHER_CHANNELS].std(axis=0, dtype=np.float64)
        # The noise streams in the order convert() uses them
        seeds = np.random.SeedSequence(cb.NOISE_SEED).spawn(5)
        for index, (directory, stem, scale) in enumerate(
            (
                ("train", "sources", source_scale),
                ("train", "targets_PI_octdec", target_scale),
                ("test", "targets_PI_year2", target_scale),
            )
        ):
            with self.subTest(file=stem):
                clean = _load(self.out / directory / f"{stem}.npz")[..., cb.WEATHER_CHANNELS]
                noisy = _load(self.out / directory / f"{stem}_noisy_weather.npz")
                draws = np.random.default_rng(seeds[index]).standard_normal(clean.shape, dtype=np.float32)
                # Recover the scale from entries that were not clipped and whose draw is not tiny
                usable = (noisy > 0) & (np.abs(draws) > 0.1)
                recovered = (noisy.astype(np.float64) - clean) / draws
                expected = np.broadcast_to(scale[None], clean.shape)
                np.testing.assert_allclose(recovered[usable], expected[usable], rtol=1e-3)
        # The two target files really span different weather: their own stds differ from the shared scale
        octdec = _load(self.out / "train" / "targets_PI_octdec.npz")[..., cb.WEATHER_CHANNELS]
        self.assertFalse(np.allclose(cb.NOISE_STD_FRACTION * octdec.std(axis=0), target_scale, rtol=1e-2))

    def test_csv_content_checks(self):
        """A CSV with a wrong header, row count or time grid, or a NaN or inf value, fails."""

        def wrong_header(frame):
            return frame.rename(columns={"u": "v"})

        def wrong_rows(frame):
            return frame.iloc[:-1]

        def bad_time(frame):
            frame["time"] = frame["time"] - cb.DELTA_T  # starts at 0 instead of 900
            return frame

        def with_value(value):
            def corrupt(frame):
                frame.iloc[3, 4] = value
                return frame

            return corrupt

        cases = {
            "header": (wrong_header, "columns"),
            "rows": (wrong_rows, "rows, expected"),
            "time": (bad_time, "time column is not"),
            "nan": (with_value(np.nan), "NaN or inf"),
            "inf": (with_value(np.inf), "NaN or inf"),
        }
        for case, (corrupt, message) in cases.items():
            with self.subTest(case=case):
                pi, rw, sources = self._bad_input(f"csv_{case}", TARGET_NAMES, TARGET_NAMES, [SOURCE_NAMES])
                path = pi / TARGET_NAMES[1]
                corrupt(pd.read_csv(path, index_col=0)).to_csv(path)
                with self.assertRaisesRegex(cb.ConversionError, message):
                    _convert(pi, rw, sources, self.root / f"csv_{case}" / "out", relax_counts=True)

    def test_source_dir_count(self):
        """Only 1 or 3 source directories are accepted."""
        pi, rw, sources = self._bad_input("two_sources", TARGET_NAMES, TARGET_NAMES, [SOURCE_NAMES, SOURCE_NAMES])
        with self.assertRaisesRegex(cb.ConversionError, "Give 1 or 3 source directories, got 2"):
            _convert(pi, rw, sources, self.root / "two_sources" / "out", relax_counts=True)

    def test_city_check(self):
        """A source city in a target directory, or a target city in a source directory, fails."""
        mixed_targets = [TARGET_NAMES[0], SOURCE_NAMES[0]]
        pi, rw, sources = self._bad_input("city_target", mixed_targets, mixed_targets, [SOURCE_NAMES])
        with self.assertRaisesRegex(cb.ConversionError, "does not belong to a target"):
            _convert(pi, rw, sources, self.root / "city_target" / "out", relax_counts=True)
        mixed_sources = [SOURCE_NAMES[0], TARGET_NAMES[0]]
        pi, rw, sources = self._bad_input("city_source", TARGET_NAMES, TARGET_NAMES, [mixed_sources])
        with self.assertRaisesRegex(cb.ConversionError, "does not belong to a source"):
            _convert(pi, rw, sources, self.root / "city_source" / "out", relax_counts=True)

    def test_count_check(self):
        """Too few files fail the count check, unless --relax-counts is given."""
        pi, rw, sources = self._bad_input("count", TARGET_NAMES[:1], TARGET_NAMES[:1], [SOURCE_NAMES])
        with self.assertRaisesRegex(cb.ConversionError, "Expected 2 files per target directory"):
            _convert(pi, rw, sources, self.root / "count" / "out")
        shapes = _convert(pi, rw, sources, self.root / "count" / "out", relax_counts=True)
        self.assertEqual(shapes["targets_PI_year2.npz"], (12, 1, 6))
        pi, rw, sources = self._bad_input("count_src", TARGET_NAMES, TARGET_NAMES, [SOURCE_NAMES[:1]])
        with self.assertRaisesRegex(cb.ConversionError, "Expected 2 files in source directory 0"):
            _convert(pi, rw, sources, self.root / "count_src" / "out")
        # The count check also passes through main's exit code
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            argv = ["--pi-dir", str(pi), "--rw-dir", str(rw), "--source-dir", str(sources[0])]
            self.assertEqual(cb.main([*argv, "--out-dir", str(self.root / "count_src" / "main")]), 1)

    def test_identical_filename_check(self):
        """PI and RW, and all source directories, must hold the same filenames, even with --relax-counts."""
        pi, rw, sources = self._bad_input("names_target", TARGET_NAMES, TARGET_NAMES[:1], [SOURCE_NAMES])
        with self.assertRaisesRegex(cb.ConversionError, "PI and RW target directories hold different filenames"):
            _convert(pi, rw, sources, self.root / "names_target" / "out", relax_counts=True)
        pi, rw, sources = self._bad_input(
            "names_source", TARGET_NAMES, TARGET_NAMES, [SOURCE_NAMES, SOURCE_NAMES[:1], SOURCE_NAMES]
        )
        with self.assertRaisesRegex(cb.ConversionError, "Source directory 1 holds different filenames"):
            _convert(pi, rw, sources, self.root / "names_source" / "out", relax_counts=True)


if __name__ == "__main__":
    unittest.main()
