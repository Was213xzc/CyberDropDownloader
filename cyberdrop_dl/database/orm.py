from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MediaItemRecord(Base):
    __tablename__ = "media_items"
    __table_args__ = (
        UniqueConstraint("domain", "url_path", "original_filename", name="uq_media_items_identity"),
        Index("ix_media_items_referer_domain_completed", "referer", "domain", "completed"),
        Index("ix_media_items_domain_album_completed", "domain", "album_id", "completed", "url_path"),
        Index("ix_media_items_download_filename", "download_filename"),
        Index("ix_media_items_domain_filename_size_completed", "domain", "download_filename", "file_size", "completed"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    domain: Mapped[str] = mapped_column(String, nullable=False)
    url_path: Mapped[str] = mapped_column(String, nullable=False)
    referer: Mapped[str] = mapped_column(String, nullable=False)
    album_id: Mapped[str | None] = mapped_column(String, nullable=True)
    download_path: Mapped[str] = mapped_column(String, nullable=False)
    download_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    original_filename: Mapped[str] = mapped_column(String, nullable=False, default="")
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    files: Mapped[list["MediaFileRecord"]] = relationship(
        back_populates="media_item",
        cascade="all, delete-orphan",
    )


class MediaFileRecord(Base):
    __tablename__ = "media_files"
    __table_args__ = (
        UniqueConstraint("folder", "download_filename", name="uq_media_files_folder_filename"),
        Index("ix_media_files_size", "file_size"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    media_item_id: Mapped[int | None] = mapped_column(ForeignKey("media_items.id"), nullable=True)
    folder: Mapped[str] = mapped_column(String, nullable=False)
    download_filename: Mapped[str] = mapped_column(String, nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    referer: Mapped[str | None] = mapped_column(String, nullable=True)
    date: Mapped[int | None] = mapped_column(Integer, nullable=True)

    media_item: Mapped[MediaItemRecord | None] = relationship(back_populates="files")
    hashes: Mapped[list["MediaHashRecord"]] = relationship(
        back_populates="file",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class MediaHashRecord(Base):
    __tablename__ = "media_hashes"
    __table_args__ = (
        UniqueConstraint("media_file_id", "hash_type", name="uq_media_hashes_file_type"),
        Index("ix_media_hashes_hash_type_hash", "hash_type", "hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    media_file_id: Mapped[int] = mapped_column(ForeignKey("media_files.id"), nullable=False)
    hash_type: Mapped[str] = mapped_column(String, nullable=False)
    hash: Mapped[str] = mapped_column(String, nullable=False)

    file: Mapped[MediaFileRecord] = relationship(back_populates="hashes")


SQLA_METADATA: Any = Base.metadata
