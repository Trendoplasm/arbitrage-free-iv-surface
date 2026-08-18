"""Command-line behaviour, including how failures are reported."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ivsurface import __version__
from ivsurface.cli import main

#: Tables the study always writes. The stress-episode table is conditional -- it holds one row per
#: named market shock inside the sample, and the short synthetic calendar reaches none of them --
#: so it is asserted separately rather than counted here.
EXPECTED_TABLES = 14
CONDITIONAL_TABLE = "stress_episodes.csv"


@pytest.fixture(scope="module")
def completed(workspace: Path) -> Path:
    """Run the study once with plots and share the output across the module.

    The pipeline calibrates every day under two regimes, so re-running it for each assertion would
    dominate the suite's runtime without testing anything new.
    """
    assert run(workspace) == 0
    return workspace / "out"


@pytest.fixture(scope="module")
def payload(completed: Path) -> dict[str, object]:
    """Return the shared run's summary."""
    loaded: dict[str, object] = json.loads((completed / "summary.json").read_text())
    return loaded


def run(workspace: Path, *extra: str) -> int:
    return main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--output-dir",
            str(workspace / "out"),
            "--start-date",
            "2011-01-07",
            "--bootstrap-iterations",
            "50",
            *extra,
        ]
    )


def test_writes_every_table_and_figure(completed: Path) -> None:
    out = completed
    written = {path.name for path in (out / "tables").glob("*.csv")}
    assert len(written) == EXPECTED_TABLES
    assert {p.name for p in (out / "plots").glob("*.png")} == {
        "observed_term_structure.png",
        "calibration_tradeoff.png",
        "fitted_surface.png",
        "correlation_history.png",
    }
    assert (out / "summary.json").exists()


def test_a_table_with_no_rows_is_skipped_rather_than_written_empty(completed: Path) -> None:
    # The synthetic calendar covers 2011 only, so no named market shock falls inside it. An empty
    # table is a result, not a failure, and writing a header-only file would misrepresent it.
    assert not (completed / "tables" / CONDITIONAL_TABLE).exists()


def test_no_plots_skips_figures(workspace: Path) -> None:
    target = workspace / "noplots"
    status = main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--output-dir",
            str(target),
            "--start-date",
            "2011-01-07",
            "--bootstrap-iterations",
            "50",
            "--no-plots",
        ]
    )
    assert status == 0
    assert not (target / "plots").exists()


def test_reports_the_headline_result(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(workspace, "--no-plots")
    output = capsys.readouterr().out
    assert "calendar violations" in output
    assert "held-out maturity" in output


def test_summary_states_the_scope(payload: dict[str, object]) -> None:
    # Every export says which dimension is observed and which is not.
    note = payload["scope_note"]
    assert isinstance(note, str)
    assert "maturity dimension" in note
    assert "strike dimension is not observed" in note


def test_the_constrained_regime_is_always_arbitrage_free(payload: dict[str, object]) -> None:
    # The study's central claim, checked on synthetic inputs so it holds structurally rather than
    # only on the particular history that happens to be downloaded.
    comparison = payload["calibration_comparison"]
    assert isinstance(comparison, list)
    constrained = next(row for row in comparison if row["no_arbitrage_enforced"])
    assert constrained["arbitrage_free_rate"] == pytest.approx(1.0)
    assert constrained["sufficient_conditions_rate"] == pytest.approx(1.0)
    assert constrained["worst_butterfly_g"] > 0


def test_enforcing_no_arbitrage_costs_fit_quality(payload: dict[str, object]) -> None:
    # The trade-off is the point of the comparison, so it is asserted rather than assumed.
    comparison = payload["calibration_comparison"]
    assert isinstance(comparison, list)
    by_regime = {row["no_arbitrage_enforced"]: row for row in comparison}
    assert by_regime[True]["skewness_rmse"] >= by_regime[False]["skewness_rmse"]


def test_the_held_out_maturity_is_configurable(workspace: Path) -> None:
    assert run(workspace, "--no-plots", "--held-out-maturity", "VIX3M") == 0
    payload = json.loads((workspace / "out" / "summary.json").read_text())
    assert payload["configuration"]["held_out_maturity"] == "VIX3M"


def test_an_unknown_held_out_maturity_is_a_usage_error(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(workspace, "--held-out-maturity", "NOTANINDEX") == 2
    assert "must be one of" in capsys.readouterr().err


def test_quiet_suppresses_progress_output(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(workspace, "--quiet", "--no-plots")
    assert capsys.readouterr().out == ""


def test_missing_input_reports_a_message_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["--data-dir", str(tmp_path / "absent"), "--output-dir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert status == 1
    assert captured.err.lower().startswith("error:")
    assert "Traceback" not in captured.err


def test_nonpositive_bootstrap_is_a_usage_error(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--output-dir",
            str(workspace / "out"),
            "--bootstrap-iterations",
            "0",
        ]
    )
    assert status == 2
    assert "must be positive" in capsys.readouterr().err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
