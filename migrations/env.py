"""Configure Alembic to use our database and table definitions."""

from logging.config import fileConfig

from alembic import context

from delivery_intelligence_platform.database import (
    make_engine,
    metadata,
)


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def run_migrations():
    engine = make_engine()

    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise RuntimeError(
        "Run migrations with a live database connection."
    )

run_migrations()