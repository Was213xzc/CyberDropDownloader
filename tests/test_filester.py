from unittest import mock

import pytest
from bs4 import BeautifulSoup

from cyberdrop_dl.crawlers.filester import FilesterCrawler
from cyberdrop_dl.data_structures.url_objects import AbsoluteHttpURL, ScrapeItem


@pytest.mark.asyncio
async def test_filester_file_allows_missing_sha256() -> None:
    crawler = FilesterCrawler(mock.Mock())
    soup = BeautifulSoup(
        """
        <html>
            <head><meta property="og:title" content="3024x4032_a83f4d5e1a210657d619431a9d6c39ca.jpg"></head>
            <body>
                <div id="detailsContent">
                    <span>Type</span><span>image/jpeg</span>
                    <span>Uploaded</span><span>2026-05-29T00:00:00Z</span>
                </div>
            </body>
        </html>
        """,
        "html.parser",
    )

    crawler.check_complete_from_referer = mock.AsyncMock(return_value=False)
    crawler.check_complete_by_hash = mock.AsyncMock(return_value=False)
    crawler.request_soup = mock.AsyncMock(return_value=soup)
    crawler._request_download = mock.AsyncMock(return_value=AbsoluteHttpURL("https://cache1.filester.me/d/token"))
    crawler.handle_file = mock.AsyncMock()

    scrape_item = ScrapeItem(url=AbsoluteHttpURL("https://filester.me/d/Mg1NRAg"))
    await crawler.file(scrape_item, "Mg1NRAg")

    crawler.check_complete_by_hash.assert_not_awaited()
    crawler._request_download.assert_awaited_once_with("Mg1NRAg")
    crawler.handle_file.assert_awaited_once()
