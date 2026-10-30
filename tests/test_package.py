import pytest

import drift
from drift.cli import main


def test_version() -> None:
    assert isinstance(drift.__version__, str)


def test_cli_help_and_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ok:
        main(["--help"])
    assert ok.value.code == 0 and "bank build" in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing:
        main([])  # a subcommand is required
    assert missing.value.code == 2
