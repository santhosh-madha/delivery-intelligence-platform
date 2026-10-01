"""Record provenance without assuming existing requests were real."""
from alembic import op
import sqlalchemy as sa

revision = "c42a001"
down_revision = "8bc4a9f5d424"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("predictions", sa.Column(
        "data_source", sa.String(16), nullable=False,
        server_default="unknown",
    ))
    op.create_check_constraint(
        "ck_predictions_data_source", "predictions",
        "data_source IN ('unknown', 'demo', 'real')",
    )


def downgrade():
    op.drop_constraint("ck_predictions_data_source", "predictions", type_="check")
    op.drop_column("predictions", "data_source")
