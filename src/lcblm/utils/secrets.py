import os
from typing import Any, Literal

SUPPORTED_PLATFORMS = ["kaggle", "local"]


def get_secrets(
    platform: Literal["kaggle", "local"],
    secrets: list[str],
) -> dict[str, Any]:
    """Get secrets from the platform's secrets utility.

    Args:
        platform: the platform this code is running on.
        secrets: list of secrets to retrieve.

    Returns:
        Dictionary mapping secret name to secret value.

    Raises:
        ValueError: if unsupported platform is passed.

    """
    match platform:
        case "kaggle":
            from better_kaggle_secrets import UserSecretsClient  # noqa: PLC0415

            user_secrets = UserSecretsClient()
            return {secret: user_secrets.get_secret(secret) for secret in secrets}
        case "local":
            from dotenv import load_dotenv  # noqa: PLC0415

            load_dotenv()
            return {secret: os.getenv(secret) for secret in secrets}
        case _:
            raise ValueError(
                f"Supported platforms are {SUPPORTED_PLATFORMS}, got {platform}",
            )
