"""Start the LogHawk API using its configured Uvicorn bind URL."""

from urllib.parse import urlparse

import uvicorn

from loghawk import config


def main() -> None:
    bind = urlparse(config.LH_API_SERVER_BIND_URL)
    uvicorn.run(
        "loghawk.api.main:app",
        app_dir=str(config.PROJECT_ROOT / "src"),
        host=bind.hostname,
        port=bind.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
