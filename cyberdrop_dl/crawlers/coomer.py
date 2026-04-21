from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from cyberdrop_dl.data_structures.url_objects import AbsoluteHttpURL
from cyberdrop_dl.exceptions import NoExtensionError

from .kemono import KemonoBaseCrawler, _thumbnail_to_src

_CDN_HOSTS = ("n1.coomer.st", "n2.coomer.st", "n3.coomer.st", "n4.coomer.st")


class CoomerCrawler(KemonoBaseCrawler):
    PRIMARY_URL: ClassVar[AbsoluteHttpURL] = AbsoluteHttpURL("https://coomer.st")
    DOMAIN: ClassVar[str] = "coomer"
    API_ENTRYPOINT = AbsoluteHttpURL("https://coomer.st/api/v1")
    SERVICES = "onlyfans", "fansly", "candfans"
    OLD_DOMAINS: ClassVar[tuple[str, ...]] = "coomer.party", "coomer.su"
    _DOWNLOAD_SLOTS: ClassVar[int | None] = 1
    _USE_DOWNLOAD_SERVERS_LOCKS: ClassVar[bool] = True

    @property
    def session_cookie(self) -> str:
        return self.manager.config_manager.authentication_data.coomer.session

    async def handle_direct_link(self, scrape_item, url: AbsoluteHttpURL | None = None) -> None:
        scrape_item.url = _thumbnail_to_src(scrape_item.url)
        link = _thumbnail_to_src(url or scrape_item.url)
        hash_value = Path(link.name).stem
        if await self.check_complete_by_hash(link, "sha256", hash_value):
            return

        try:
            filename, ext = self.get_filename_and_ext(link.query.get("f") or link.name)
        except NoExtensionError:
            filename, ext = self.get_filename_and_ext(link.name)

        await self.handle_file(
            link,
            scrape_item,
            link.name,
            ext,
            custom_filename=filename,
            fallbacks=_coomer_download_fallbacks(link),
        )


def _coomer_download_fallbacks(url: AbsoluteHttpURL) -> list[AbsoluteHttpURL]:
    candidates: list[AbsoluteHttpURL] = []
    seen = {url}

    primary = url.with_host("coomer.st")
    if primary not in seen:
        candidates.append(primary)
        seen.add(primary)

    for host in _CDN_HOSTS:
        candidate = url.with_host(host)
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    return candidates
