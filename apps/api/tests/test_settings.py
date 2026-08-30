from app.core.settings import Settings, get_settings


def test_rewrites_railway_postgres_url() -> None:
    settings = Settings(
        database_url="postgresql://starter:starter@localhost:5432/starter",
    )
    assert settings.database_url == "postgresql+psycopg://starter:starter@localhost:5432/starter"


def test_rewrites_postgres_scheme() -> None:
    settings = Settings(database_url="postgres://user:pass@db:5432/app")
    assert settings.database_url == "postgresql+psycopg://user:pass@db:5432/app"


def test_leaves_explicit_psycopg_url() -> None:
    url = "postgresql+psycopg://starter:starter@localhost:5432/starter"
    assert Settings(database_url=url).database_url == url


def test_rerank_defaults_off() -> None:
    assert Settings().rerank_enabled is False


def test_get_settings_uses_cache_clear() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
