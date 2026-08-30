from alembic.config import Config

from app.core.db import alembic_sqlalchemy_url, connect_args_for


def test_postgres_connect_args_include_timeout() -> None:
    assert connect_args_for("postgresql+psycopg://starter:starter@localhost:5432/starter") == {
        "connect_timeout": 5
    }


def test_sqlite_connect_args_stay_local() -> None:
    assert connect_args_for("sqlite://") == {"check_same_thread": False}


def test_alembic_url_escapes_percent_for_configparser() -> None:
    url = "postgresql+psycopg://starter:p%40ss%25w@localhost:5432/starter"
    escaped = alembic_sqlalchemy_url(url)
    config = Config()
    config.set_main_option("sqlalchemy.url", escaped)
    assert config.get_main_option("sqlalchemy.url") == url


def test_alembic_url_leaves_plain_passwords() -> None:
    url = "postgresql+psycopg://starter:starter@localhost:5432/starter"
    assert alembic_sqlalchemy_url(url) == url
