import sys

import pytest

from benchmark import CommandExecutionError, run_command


def test_run_command_timeout_raises_error():
    cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
    with pytest.raises(CommandExecutionError) as exc:
        run_command(cmd, timeout_seconds=1, context="timeout_case")
    assert "超时" in str(exc.value)
    assert "timeout_case" in str(exc.value)
