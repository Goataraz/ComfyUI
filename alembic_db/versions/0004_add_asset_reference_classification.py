"""
Add persisted asset classification columns.

Revision ID: 0004_add_asset_reference_classification
Revises: 0003_add_metadata_job_id
Create Date: 2026-05-29
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_add_asset_reference_classification"
down_revision = "0003_add_metadata_job_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.add_column(sa.Column("asset_type", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("model_folder", sa.String(length=512), nullable=True))
        batch_op.create_index("ix_asset_references_asset_type", ["asset_type"])
        batch_op.create_index("ix_asset_references_model_folder", ["model_folder"])


def downgrade() -> None:
    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.drop_index("ix_asset_references_model_folder")
        batch_op.drop_index("ix_asset_references_asset_type")
        batch_op.drop_column("model_folder")
        batch_op.drop_column("asset_type")
