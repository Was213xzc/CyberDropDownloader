from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_database_modernization"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "media_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("domain", sa.String(), nullable=False),
        sa.Column("url_path", sa.String(), nullable=False),
        sa.Column("referer", sa.String(), nullable=False),
        sa.Column("album_id", sa.String(), nullable=True),
        sa.Column("download_path", sa.String(), nullable=False),
        sa.Column("download_filename", sa.String(), nullable=True),
        sa.Column("original_filename", sa.String(), nullable=False, server_default=""),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("domain", "url_path", "original_filename", name="uq_media_items_identity"),
    )
    op.create_index(
        "ix_media_items_referer_domain_completed",
        "media_items",
        ["referer", "domain", "completed"],
        unique=False,
    )
    op.create_index(
        "ix_media_items_domain_album_completed",
        "media_items",
        ["domain", "album_id", "completed", "url_path"],
        unique=False,
    )
    op.create_index("ix_media_items_download_filename", "media_items", ["download_filename"], unique=False)
    op.create_index(
        "ix_media_items_domain_filename_size_completed",
        "media_items",
        ["domain", "download_filename", "file_size", "completed"],
        unique=False,
    )

    op.create_table(
        "media_files",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("media_item_id", sa.Integer(), sa.ForeignKey("media_items.id"), nullable=True),
        sa.Column("folder", sa.String(), nullable=False),
        sa.Column("download_filename", sa.String(), nullable=False),
        sa.Column("original_filename", sa.String(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("referer", sa.String(), nullable=True),
        sa.Column("date", sa.Integer(), nullable=True),
        sa.UniqueConstraint("folder", "download_filename", name="uq_media_files_folder_filename"),
    )
    op.create_index("ix_media_files_size", "media_files", ["file_size"], unique=False)

    op.create_table(
        "media_hashes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("media_file_id", sa.Integer(), sa.ForeignKey("media_files.id"), nullable=False),
        sa.Column("hash_type", sa.String(), nullable=False),
        sa.Column("hash", sa.String(), nullable=False),
        sa.UniqueConstraint("media_file_id", "hash_type", name="uq_media_hashes_file_type"),
    )
    op.create_index("ix_media_hashes_hash_type_hash", "media_hashes", ["hash_type", "hash"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_media_hashes_hash_type_hash", table_name="media_hashes")
    op.drop_table("media_hashes")
    op.drop_index("ix_media_files_size", table_name="media_files")
    op.drop_table("media_files")
    op.drop_index("ix_media_items_domain_filename_size_completed", table_name="media_items")
    op.drop_index("ix_media_items_download_filename", table_name="media_items")
    op.drop_index("ix_media_items_domain_album_completed", table_name="media_items")
    op.drop_index("ix_media_items_referer_domain_completed", table_name="media_items")
    op.drop_table("media_items")
