from __future__ import annotations

from cyberdrop_dl.exceptions import ErrorLogMessage


class FakeUnknownError(Exception):
    pass


def test_unknown_curl_timeout_is_classified_as_timeout() -> None:
    error = FakeUnknownError("Failed to perform, curl: (28) Connection timed out after 15001 milliseconds.")

    result = ErrorLogMessage.from_unknown_exc(error)

    assert result.ui_failure == "Timeout"
    assert result.csv_log_msg == "Timeout"
    assert "curl: (28)" in result.main_log_msg


def test_windows_file_exists_collision_is_classified_clearly() -> None:
    error = FakeUnknownError(
        "[WinError 183] Cannot create a file when that file already exists: "
        "'E:\\video compression\\Downloads\\foo.mp4.part' -> "
        "'E:\\video compression\\Downloads\\foo.mp4'"
    )

    result = ErrorLogMessage.from_unknown_exc(error)

    assert result.ui_failure == "Destination File Exists"
    assert result.csv_log_msg == "Destination File Exists"
    assert "WinError 183" in result.main_log_msg


def test_unclassified_unknown_error_still_points_to_logs() -> None:
    result = ErrorLogMessage.from_unknown_exc(FakeUnknownError("Something odd happened"))

    assert result.ui_failure == "Unknown"
    assert result.csv_log_msg == "See logs for details"
