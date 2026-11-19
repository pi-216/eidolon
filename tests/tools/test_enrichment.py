import os
import pytest
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, exc

from tools.hegre_dataset.enrichment import generate_image_list, run_stratum_enrichment
from tools.hegre_dataset.dataset import HegreDataset


def _make_test_db(db_path, rows, with_distances=False):
    """Create a test SQLite DB with the images table and insert rows.

    Args:
        rows: List of tuples. If with_distances=True, tuples are
            (image_path, status, zg_distance, af_distance).
            Otherwise, tuples are (image_path, status).
    """
    conn = sqlite3.connect(str(db_path))
    if with_distances:
        conn.execute("CREATE TABLE IF NOT EXISTS images "
                     "(image_path TEXT, status TEXT, zg_distance REAL, af_distance REAL)")
    else:
        conn.execute("CREATE TABLE IF NOT EXISTS images (image_path TEXT, status TEXT)")
    conn.executemany("INSERT INTO images VALUES ({})".format(
        "?, ?" if not with_distances else "?, ?, ?, ?"), rows)
    conn.commit()
    conn.close()


def _make_ds(tmp_path, rows, with_distances=False):
    """Create a test HegreDataset and return (ds, root).

    The DB lives OUTSIDE the dataset root and is injected as an engine: the
    root must stay clean so the fail-closed review.db guard keeps firing,
    and no test may reach the project's PostgreSQL.
    """
    root = tmp_path / "dataset"
    root.mkdir()
    db_path = tmp_path / "hegre_test.db"
    _make_test_db(db_path, rows, with_distances=with_distances)
    engine = create_engine(f"sqlite:///{db_path}")
    return HegreDataset(root, engine=engine), root


def _make_empty_ds_env(tmp_path, monkeypatch):
    """Point EIDOLON_DB_URL at a schema-only SQLite DB in a clean root."""
    root = tmp_path / "dataset"
    root.mkdir()
    db_path = tmp_path / "hegre_test.db"
    _make_test_db(db_path, [])
    monkeypatch.setenv("EIDOLON_DB_URL", f"sqlite:///{db_path}")
    return root


def test_generate_image_list_default_both(tmp_path):
    """Default status_filter='both' returns approved + unreviewed only."""
    ds, root = _make_ds(tmp_path, [
        ("app1.jpg", "approved"),
        ("rej1.jpg", "rejected"),
        ("app2.jpg", "approved"),
        ("pend1.jpg", "pending"),
        ("unr1.jpg", "unreviewed"),
        ("unr2.jpg", "unreviewed"),
    ])
    for f in ["app1.jpg", "rej1.jpg", "app2.jpg", "pend1.jpg", "unr1.jpg", "unr2.jpg"]:
        (root / f).touch()

    paths = generate_image_list(ds)

    assert len(paths) == 4
    path_strs = {str(p) for p in paths}
    for ok in ["app1.jpg", "app2.jpg", "unr1.jpg", "unr2.jpg"]:
        assert str((root / ok).absolute()) in path_strs
    for bad in ["rej1.jpg", "pend1.jpg"]:
        assert str((root / bad).absolute()) not in path_strs


def test_generate_image_list_approved_only(tmp_path):
    """status_filter='approved' returns approved only."""
    ds, root = _make_ds(tmp_path, [
        ("app1.jpg", "approved"),
        ("unr1.jpg", "unreviewed"),
        ("app2.jpg", "approved"),
        ("tainted.jpg", "tainted:extraction_nonface"),
    ])
    for f in ["app1.jpg", "unr1.jpg", "app2.jpg", "tainted.jpg"]:
        (root / f).touch()

    paths = generate_image_list(ds, status_filter="approved")

    assert len(paths) == 2
    path_strs = {str(p) for p in paths}
    assert str((root / "app1.jpg").absolute()) in path_strs
    assert str((root / "app2.jpg").absolute()) in path_strs
    assert str((root / "unr1.jpg").absolute()) not in path_strs


def test_generate_image_list_unreviewed_only(tmp_path):
    """status_filter='unreviewed' returns unreviewed only."""
    ds, root = _make_ds(tmp_path, [
        ("app1.jpg", "approved"),
        ("unr1.jpg", "unreviewed"),
        ("unr2.jpg", "unreviewed"),
        ("rej1.jpg", "rejected"),
    ])
    for f in ["app1.jpg", "unr1.jpg", "unr2.jpg", "rej1.jpg"]:
        (root / f).touch()

    paths = generate_image_list(ds, status_filter="unreviewed")

    assert len(paths) == 2
    path_strs = {str(p) for p in paths}
    assert str((root / "unr1.jpg").absolute()) in path_strs
    assert str((root / "unr2.jpg").absolute()) in path_strs


def test_generate_image_list_empty(tmp_path):
    """Returns empty list when no matching images exist."""
    ds, root = _make_ds(tmp_path, [
        ("rej1.jpg", "rejected"),
        ("tainted.jpg", "tainted:extraction_nonface"),
    ])
    for f in ["rej1.jpg", "tainted.jpg"]:
        (root / f).touch()

    paths = generate_image_list(ds)
    assert len(paths) == 0


def test_reviewdb_guard_fires_on_stale_sqlite(tmp_path):
    """A review.db at the dataset root must still raise — the guard stays armed."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "review.db").touch()
    try:
        HegreDataset(root, engine=create_engine("sqlite:///:memory:"))
    except RuntimeError as e:
        assert "review.db still exists" in str(e)
    else:
        raise AssertionError("fail-closed guard did not fire")


def test_run_stratum_enrichment_no_images(tmp_path, monkeypatch):
    """When generate_image_list returns empty, enrichment skips early."""
    root = _make_empty_ds_env(tmp_path, monkeypatch)
    (root / "faces").mkdir()
    (root / "stratum").mkdir()

    with patch("subprocess.run") as mock_run:
        run_stratum_enrichment(root, root / "hegre_test.db", root,
                               status_filter="approved")
        mock_run.assert_not_called()


def test_cli_enrich_status_flag(tmp_path):
    """The --status flag is accepted by the enrich CLI subparser."""
    root = tmp_path / "dataset"
    root.mkdir()
    db_path = tmp_path / "hegre_test.db"
    _make_test_db(db_path, [])

    result = subprocess.run(
        [sys.executable, "-m", "tools.hegre_dataset", "enrich",
         "--dataset", str(root), "--status", "approved", "--skip-stratum"],
        capture_output=True, text=True,
        env={**os.environ, "EIDOLON_DB_URL": f"sqlite:///{db_path}"},
    )
    assert result.returncode == 0
    assert "approved" in result.stdout.lower()


class TestZgMaxDistanceFilter:
    """Tests for the --zg-max-distance filter on enrich."""

    def test_zg_filter_excludes_high_distance_approved(self, tmp_path):
        """Approved images with zg_distance > threshold are excluded."""
        ds, root = _make_ds(tmp_path, [
            ("good.jpg", "approved", 10.0, None),
            ("bad.jpg", "approved", 200.0, None),
            ("none_yet.jpg", "approved", None, None),
        ], with_distances=True)
        for f in ["good.jpg", "bad.jpg", "none_yet.jpg"]:
            (root / f).touch()

        paths = generate_image_list(ds, status_filter="approved",
                                     zg_max_distance=100.0)

        path_strs = {str(p) for p in paths}
        assert str((root / "good.jpg").absolute()) in path_strs
        assert str((root / "none_yet.jpg").absolute()) in path_strs
        assert str((root / "bad.jpg").absolute()) not in path_strs

    def test_zg_filter_does_not_affect_unreviewed(self, tmp_path):
        """zg_max_distance filter does not apply to unreviewed images."""
        ds, root = _make_ds(tmp_path, [
            ("unr_good.jpg", "unreviewed", 5.0, None),
            ("unr_bad.jpg", "unreviewed", 500.0, None),
        ], with_distances=True)
        for f in ["unr_good.jpg", "unr_bad.jpg"]:
            (root / f).touch()

        paths = generate_image_list(ds, status_filter="unreviewed",
                                     zg_max_distance=100.0)

        assert len(paths) == 2

    def test_zg_filter_both_mode(self, tmp_path):
        """In 'both' mode, only approved images are filtered by zg_distance."""
        ds, root = _make_ds(tmp_path, [
            ("app_good.jpg", "approved", 10.0, None),
            ("app_bad.jpg", "approved", 300.0, None),
            ("unr_ok.jpg", "unreviewed", 500.0, None),
        ], with_distances=True)
        for f in ["app_good.jpg", "app_bad.jpg", "unr_ok.jpg"]:
            (root / f).touch()

        paths = generate_image_list(ds, status_filter="both",
                                     zg_max_distance=100.0)

        path_strs = {str(p) for p in paths}
        assert str((root / "app_good.jpg").absolute()) in path_strs
        assert str((root / "unr_ok.jpg").absolute()) in path_strs
        assert str((root / "app_bad.jpg").absolute()) not in path_strs

    def test_zg_filter_requires_column(self, tmp_path):
        """Without a zg_distance column the filter query fails loudly, not silently."""
        ds, root = _make_ds(tmp_path, [
            ("img1.jpg", "approved"),
            ("img2.jpg", "approved"),
        ])
        for f in ["img1.jpg", "img2.jpg"]:
            (root / f).touch()

        with pytest.raises(exc.OperationalError):
            generate_image_list(ds, status_filter="approved",
                                 zg_max_distance=50.0)


class TestSortBy:
    """Tests for the --sort-by parameter on enrich."""

    def test_sort_by_af_ascending(self, tmp_path):
        """--sort-by af returns images ordered by af_distance ASC."""
        ds, root = _make_ds(tmp_path, [
            ("img1.jpg", "approved", 5.0, 0.3),
            ("img2.jpg", "approved", 10.0, 0.1),
            ("img3.jpg", "approved", 1.0, None),
        ], with_distances=True)
        for f in ["img1.jpg", "img2.jpg", "img3.jpg"]:
            (root / f).touch()

        paths = generate_image_list(ds, status_filter="approved",
                                     sort_by="af")

        stems = [Path(str(p)).stem for p in paths]
        assert stems == ["img2", "img1", "img3"], f"Got {stems}"

    def test_sort_by_zg_ascending(self, tmp_path):
        """--sort-by zg returns images ordered by zg_distance ASC."""
        ds, root = _make_ds(tmp_path, [
            ("img1.jpg", "approved", 5.0, None),
            ("img2.jpg", "approved", 1.0, None),
            ("img3.jpg", "approved", None, None),
        ], with_distances=True)
        for f in ["img1.jpg", "img2.jpg", "img3.jpg"]:
            (root / f).touch()

        paths = generate_image_list(ds, status_filter="approved",
                                     sort_by="zg")

        stems = [Path(str(p)).stem for p in paths]
        assert stems == ["img2", "img1", "img3"], f"Got {stems}"

    def test_cli_sort_by_flag(self, tmp_path):
        """--sort-by is accepted by the CLI parser."""
        root = tmp_path / "dataset"
        root.mkdir()
        db_path = tmp_path / "hegre_test.db"
        _make_test_db(db_path, [], with_distances=True)

        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", "enrich",
             "--dataset", str(root), "--sort-by", "af", "--skip-stratum"],
            capture_output=True, text=True,
            env={**os.environ, "EIDOLON_DB_URL": f"sqlite:///{db_path}"},
        )
        assert result.returncode == 0
        assert "--sort-by" not in (result.stderr or "")

    def test_cli_zg_max_distance_flag(self, tmp_path):
        """--zg-max-distance is accepted by the CLI parser."""
        root = tmp_path / "dataset"
        root.mkdir()
        db_path = tmp_path / "hegre_test.db"
        _make_test_db(db_path, [], with_distances=True)

        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", "enrich",
             "--dataset", str(root), "--zg-max-distance", "100.0", "--skip-stratum"],
            capture_output=True, text=True,
            env={**os.environ, "EIDOLON_DB_URL": f"sqlite:///{db_path}"},
        )
        assert result.returncode == 0
        assert "zg" in result.stdout.lower() or result.returncode == 0
