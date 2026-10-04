"""Apply the SQL files in infra/neon to a Postgres database, each once, in name order.

Usage:  G2M_DATABASE_URL=postgresql://... python -m genome2mic.api.services.migrations [migrations_dir]

Apply to a Neon test branch first, then to production. Applied files are recorded in
schema_migrations, so re-running skips them. Never edit an applied file; add a new one.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

DEFAULT_MIGRATIONS_DIR = Path("infra/neon")


async def apply_migrations(database_url: str, migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> list[str]:
    """Apply every *.sql file not yet recorded. Returns the names applied this run."""
    files = sorted(migrations_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no .sql files in {migrations_dir}")
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(
            "create table if not exists schema_migrations ("
            " name text primary key, applied_at timestamptz not null default now())"
        )
        done = {row["name"] for row in await conn.fetch("select name from schema_migrations")}
        applied = []
        for path in files:
            if path.name in done:
                logger.info("skip %s (already applied)", path.name)
                continue
            async with conn.transaction():
                await conn.execute(path.read_text(encoding="utf-8"))
                await conn.execute("insert into schema_migrations (name) values ($1)", path.name)
            logger.info("applied %s", path.name)
            applied.append(path.name)
        return applied
    finally:
        await conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    database_url = os.environ.get("G2M_DATABASE_URL")
    if not database_url:
        sys.exit("G2M_DATABASE_URL is not set")
    migrations_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MIGRATIONS_DIR
    applied = asyncio.run(apply_migrations(database_url, migrations_dir))
    logger.info("%d migration(s) applied", len(applied))


if __name__ == "__main__":
    main()
