from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cyberdrop_dl.database import RetryMediaRow
from cyberdrop_dl.data_structures.url_objects import AbsoluteHttpURL, ScrapeItem
from cyberdrop_dl.scraper import scrape_mapper
from cyberdrop_dl.scraper.scrape_mapper import _create_item_from_row
from cyberdrop_dl.utils.utilities import parse_url

_MOCK_ROW = RetryMediaRow(
    referer="https://drive.google.com/file/d/1F0YBsnQRvrMbK0p9UlnyLu88kqQ0j_F6/edit",
    download_path="/cdl/downloads",
    completed_at=None,
    created_at=None,
)


@pytest.fixture
def item() -> ScrapeItem:
    return ScrapeItem(url=AbsoluteHttpURL("https://drive.google.com"))


@pytest.fixture
def row() -> RetryMediaRow:
    return RetryMediaRow(
        referer=_MOCK_ROW.referer,
        download_path=_MOCK_ROW.download_path,
        completed_at=_MOCK_ROW.completed_at,
        created_at=_MOCK_ROW.created_at,
    )


@pytest.fixture
def row_with_dates() -> RetryMediaRow:
    return RetryMediaRow(
        referer=_MOCK_ROW.referer,
        download_path=_MOCK_ROW.download_path,
        completed_at=datetime.now(),
        created_at=datetime(2023, 1, 1, 10, 0, 0),
    )


def test_scrape_item_creation(row: RetryMediaRow) -> None:
    item = _create_item_from_row(row)
    assert isinstance(item, ScrapeItem)
    assert item.url == AbsoluteHttpURL("https://drive.google.com/file/d/1F0YBsnQRvrMbK0p9UlnyLu88kqQ0j_F6/edit")
    assert item.retry_path == Path("/cdl/downloads")
    assert item.part_of_album is True
    assert item.completed_at is None
    assert item.created_at is None


def test_item_with_completed_at(row_with_dates) -> None:
    completed_at = row_with_dates.completed_at
    row = RetryMediaRow(
        referer=row_with_dates.referer,
        download_path=row_with_dates.download_path,
        completed_at=completed_at,
        created_at=None,
    )

    item = _create_item_from_row(row)
    expected_timestamp = int(completed_at.timestamp())
    assert item.completed_at == expected_timestamp
    assert item.created_at is None


def test_item_with_created_at(row) -> None:
    now = datetime.now()
    row = RetryMediaRow(
        referer=row.referer,
        download_path=row.download_path,
        completed_at=row.completed_at,
        created_at=now,
    )

    item = _create_item_from_row(row)
    assert item.created_at == int(now.timestamp())
    assert item.completed_at is None


def test_item_with_both_dates(row_with_dates) -> None:
    item = _create_item_from_row(row_with_dates)
    expected_completed_timestamp = int(row_with_dates.completed_at.timestamp())
    expected_created_timestamp = int(row_with_dates.created_at.timestamp())
    assert item.completed_at == expected_completed_timestamp
    assert item.created_at == expected_created_timestamp


def test_missing_download_path(row) -> None:
    with pytest.raises(TypeError, match="download_path"):
        _ = RetryMediaRow(
            referer=row.referer,
            completed_at=row.completed_at,
            created_at=row.created_at,
        )


def test_invalid_date_format(row) -> None:
    row = RetryMediaRow(
        referer=row.referer,
        download_path=row.download_path,
        completed_at="invalid date",  # type: ignore[arg-type]
        created_at=row.created_at,
    )

    with pytest.raises(AttributeError):
        _create_item_from_row(row)


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://megacloud.blog/embed-2/v3/e-1/TZb4gRkOQ642?k=1&autoPlay=1&oa=0&asi=1",
            "/embed-2/v3/e-1/TZb4gRkOQ642",
        ),
        (
            "https://www.mediafire.com/file/ctppmpm7giofsgv/ADOFAI.vpk",
            "ADOFAI.vpk",
        ),
        (
            "https://mega.nz/#!Ue5VRSIQ!kC2E4a4JwfWWCWYNJovGFHlbz8F",
            "/#!Ue5VRSIQ!kC2E4a4JwfWWCWYNJovGFHlbz8F",
        ),
        (
            "https://mega.nz/folder/oZZxyBrY#oU4jASLPpJVvqGHJIMRcgQ/file/IYZABDGY",
            "/folder/oZZxyBrY#oU4jASLPpJVvqGHJIMRcgQ/file/IYZABDGY",
        ),
        (
            "https://c.bunkr-cache.se/HwdRnHMUiWOQevCg/1df93418-5063-4e1b-851e-9470cb8fc5c6.mp4",
            "/HwdRnHMUiWOQevCg/1df93418-5063-4e1b-851e-9470cb8fc5c6.mp4",
        ),
        (
            "https://e-hentai.network/h/1bb8b499a5a1a21f9e25e2c42513f310c20e83a9-115314-1280-720-wbp/keystamp=1763995200-3f6832af21;fileindex=169742365;xres=1280/1_2.webp",
            "/h/1bb8b499a5a1a21f9e25e2c42513f310c20e83a9-115314-1280-720-wbp",
        ),
        (
            "https://app.koofr.net/content/links/0a00467b-2901-4213-8d71-44fad80de82d/files/get/Cyberdrop-DL.v8.4.0.zip?path=/Cyberdrop-DL.v8.4.0.zip",
            "/content/links/0a00467b-2901-4213-8d71-44fad80de82d/files/get/Cyberdrop-DL.v8.4.0.zip?path=/Cyberdrop-DL.v8.4.0.zip",
        ),
        (
            "https://transfer.it/cs/g?x=yhWbjogXxRLL&n=qgxVBD5D&fn=start_linux.sh",
            "/cs/g?x=yhWbjogXxRLL&n=qgxVBD5D&fn=start_linux.sh",
        ),
    ],
)
def test_create_db_path(url: str, expected: str) -> None:
    crawlers = scrape_mapper.get_crawlers_mapping()
    url_ = parse_url(url)
    crawler = scrape_mapper.match_url_to_crawler(crawlers, url_)
    assert crawler
    path = crawler.create_db_path(url_)
    assert path == expected


class TestGetDownloadPath:
    def test_loose_file(self, item: ScrapeItem) -> None:
        assert not item.parent_title
        assert not item.part_of_album
        assert not item.retry_path
        download_path = item.create_download_path("cyberdrop")
        assert download_path == Path("Loose Files (cyberdrop)")

    def test_loose_file_with_parent(self, item: ScrapeItem) -> None:
        item.add_to_parent_title("a/sub/folder")
        download_path = item.create_download_path("cyberdrop")
        assert download_path == Path("a-sub-folder/Loose Files (cyberdrop)")

    def test_album_file(self, item: ScrapeItem) -> None:
        item.add_to_parent_title("a/sub/folder")
        item.part_of_album = True
        download_path = item.create_download_path("cyberdrop")
        assert download_path == Path("a-sub-folder")

    def test_retry_path(self, item: ScrapeItem) -> None:
        item.add_to_parent_title("a/sub/folder")
        item.part_of_album = True
        item.retry_path = retry_path = Path("a/retry/path")
        download_path = item.create_download_path("cyberdrop")
        assert download_path == retry_path
