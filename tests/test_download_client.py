from typing import cast

import pytest

from cyberdrop_dl.clients import download_client
from cyberdrop_dl.crawlers.coomer import _coomer_download_fallbacks
from cyberdrop_dl.data_structures import MediaItem
from cyberdrop_dl.data_structures.url_objects import AbsoluteHttpURL


def _item(fallbacks_: object) -> MediaItem:
    class Item:
        fallbacks = fallbacks_

    return cast("MediaItem", Item)  # pyright: ignore[reportInvalidCast]


def test_fallback_generator_with_none() -> None:
    item = _item(None)
    gen = download_client._fallback_generator(item)
    with pytest.raises(StopIteration):
        _ = gen.send(None)


def test_fallback_generator_with_list() -> None:
    item = _item(["url1", "url2", "url3"])
    gen = download_client._fallback_generator(item)
    assert gen.__next__() == "url1"
    assert gen.__next__() == "url2"
    assert gen.send(12345) == "url3"
    with pytest.raises(StopIteration):
        _ = gen.send(None)


def test_fallback_generator_with_generator() -> None:
    def _fallback_gen(resp: object, retry: object):
        return retry

    item = _item(_fallback_gen)
    gen = download_client._fallback_generator(item)

    assert gen.send(12345) == 1
    assert gen.send(12345) == 2
    with pytest.raises(StopIteration):
        _ = gen.send(None)


def test_connection_error_fallback_uses_next_list_url() -> None:
    item = _item([AbsoluteHttpURL("https://n1.coomer.st/data/file.jpg"), AbsoluteHttpURL("https://n2.coomer.st/data/file.jpg")])
    gen = download_client._fallback_generator(item)

    assert download_client._next_fallback_url_for_connection_error(gen) == AbsoluteHttpURL(
        "https://n1.coomer.st/data/file.jpg"
    )
    assert download_client._next_fallback_url_for_connection_error(gen) == AbsoluteHttpURL(
        "https://n2.coomer.st/data/file.jpg"
    )
    assert download_client._next_fallback_url_for_connection_error(gen) is None


def test_connection_error_fallback_does_not_use_callable_without_response() -> None:
    def _fallback_gen(resp: object, retry: object):
        return retry

    item = _item(_fallback_gen)
    gen = download_client._fallback_generator(item)
    assert download_client._next_fallback_url_for_connection_error(gen) is None


def test_coomer_download_fallbacks_include_origin_and_all_cdns() -> None:
    url = AbsoluteHttpURL("https://coomer.st/data/30/1e/file.jpg?f=name.jpg")

    assert _coomer_download_fallbacks(url) == [
        AbsoluteHttpURL("https://n1.coomer.st/data/30/1e/file.jpg?f=name.jpg"),
        AbsoluteHttpURL("https://n2.coomer.st/data/30/1e/file.jpg?f=name.jpg"),
        AbsoluteHttpURL("https://n3.coomer.st/data/30/1e/file.jpg?f=name.jpg"),
        AbsoluteHttpURL("https://n4.coomer.st/data/30/1e/file.jpg?f=name.jpg"),
    ]


def test_coomer_download_fallbacks_rotate_from_cdn_to_origin_and_others() -> None:
    url = AbsoluteHttpURL("https://n2.coomer.st/data/30/1e/file.jpg?f=name.jpg")

    assert _coomer_download_fallbacks(url) == [
        AbsoluteHttpURL("https://coomer.st/data/30/1e/file.jpg?f=name.jpg"),
        AbsoluteHttpURL("https://n1.coomer.st/data/30/1e/file.jpg?f=name.jpg"),
        AbsoluteHttpURL("https://n3.coomer.st/data/30/1e/file.jpg?f=name.jpg"),
        AbsoluteHttpURL("https://n4.coomer.st/data/30/1e/file.jpg?f=name.jpg"),
    ]
