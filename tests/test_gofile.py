from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from cyberdrop_dl.crawlers.gofile import GoFileCrawler


class _ResponseContext:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response

    async def __aenter__(self) -> SimpleNamespace:
        return self.response

    async def __aexit__(self, *_: object) -> None:
        return None


def _manager(api_key: str = "") -> SimpleNamespace:
    return SimpleNamespace(auth_config=SimpleNamespace(gofile=SimpleNamespace(api_key=api_key)))


def _gofile_file(name: str, mimetype: str | None = None) -> dict[str, object]:
    file: dict[str, object] = {
        "canAccess": True,
        "createTime": 1,
        "id": "file-id",
        "link": "https://store9.gofile.io/download/web/file-id/clip",
        "md5": "d41d8cd98f00b204e9800998ecf8427e",
        "name": name,
        "type": "file",
    }
    if mimetype:
        file["mimetype"] = mimetype
    return file


def _response(content_type: str, data: bytes = b"", filename: str | None = None) -> SimpleNamespace:
    response = SimpleNamespace(content_type=content_type, read=mock.AsyncMock(return_value=data))
    if filename:
        response.filename = filename
    return response


async def test_gofile_startup_continues_with_configured_api_key_when_website_token_fails() -> None:
    crawler = GoFileCrawler(_manager("premium-token"))
    crawler.request_text = mock.AsyncMock(side_effect=TimeoutError)
    crawler.update_cookies = mock.Mock()
    crawler.log = mock.Mock()

    await crawler._get_credentials(crawler.parse_url("https://api.gofile.io"))

    assert crawler.headers == {"Authorization": "Bearer premium-token"}
    crawler.update_cookies.assert_called_once_with({"accountToken": "premium-token"})
    crawler.log.assert_called_once()


async def test_gofile_extensionless_name_uses_normalized_mimetype() -> None:
    crawler = GoFileCrawler(_manager())
    crawler.request = mock.Mock()
    link = crawler.parse_url("https://store9.gofile.io/download/web/file-id/clip")

    filename, ext = await crawler._get_filename_and_ext(_gofile_file("Clip Title", "video/mp4; charset=binary"), link)

    assert filename == "Clip Title.mp4"
    assert ext == ".mp4"
    crawler.request.assert_not_called()


async def test_gofile_extensionless_name_uses_response_content_type() -> None:
    crawler = GoFileCrawler(_manager())
    crawler.request = mock.Mock(return_value=_ResponseContext(_response("video/mp4; charset=binary")))
    link = crawler.parse_url("https://store9.gofile.io/download/web/file-id/clip")

    filename, ext = await crawler._get_filename_and_ext(_gofile_file("Clip Title"), link)

    assert filename == "Clip Title.mp4"
    assert ext == ".mp4"
    crawler.request.assert_called_once_with(link, method="HEAD", headers=None, cache_disabled=True)


async def test_gofile_extensionless_name_uses_response_filename_extension() -> None:
    crawler = GoFileCrawler(_manager())
    crawler.request = mock.Mock(return_value=_ResponseContext(_response("application/octet-stream", filename="server-name.mp4")))
    link = crawler.parse_url("https://store9.gofile.io/download/web/file-id/clip")

    filename, ext = await crawler._get_filename_and_ext(_gofile_file("Clip Title"), link)

    assert filename == "Clip Title.mp4"
    assert ext == ".mp4"


async def test_gofile_extensionless_name_ignores_html_header_and_sniffs_mp4_bytes() -> None:
    crawler = GoFileCrawler(_manager())
    header = b"\x00\x00\x00\x1cftypM4V \x00\x00\x00\x01isomavc1mp42"
    crawler.request = mock.Mock(
        side_effect=[
            _ResponseContext(_response("text/html; charset=utf-8")),
            _ResponseContext(_response("text/html; charset=utf-8", data=header)),
        ]
    )
    link = crawler.parse_url("https://store9.gofile.io/download/web/file-id/clip")

    filename, ext = await crawler._get_filename_and_ext(_gofile_file("Clip Title"), link)

    assert filename == "Clip Title.mp4"
    assert ext == ".mp4"


async def test_gofile_extension_probe_uses_auth_headers_and_sufficient_range() -> None:
    crawler = GoFileCrawler(_manager())
    crawler.headers = {"Authorization": "Bearer premium-token"}
    header = b"\x00\x00\x00\x1cftypM4V \x00\x00\x00\x01isomavc1mp42"
    crawler.request = mock.Mock(
        side_effect=[
            _ResponseContext(_response("text/html; charset=utf-8")),
            _ResponseContext(_response("text/html; charset=utf-8", data=header)),
        ]
    )
    link = crawler.parse_url("https://store9.gofile.io/download/web/file-id/clip")

    filename, ext = await crawler._get_filename_and_ext(_gofile_file("Clip Title"), link)

    assert filename == "Clip Title.mp4"
    assert ext == ".mp4"
    assert crawler.request.mock_calls == [
        mock.call(
            link,
            method="HEAD",
            headers={"Authorization": "Bearer premium-token"},
            cache_disabled=True,
        ),
        mock.call(
            link,
            method="GET",
            headers={"Authorization": "Bearer premium-token", "Range": "bytes=0-63"},
            cache_disabled=True,
        ),
    ]
