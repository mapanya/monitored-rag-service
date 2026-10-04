"""Central settings for the whole project.

Every value comes from the .env file in the project folder, so secrets
and settings live in one place and never inside the code. If a value is
missing from .env, a sensible default is used where one is safe.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Reads the .env file in the project folder and makes its values
# available to os.getenv() below. If there is no .env file this does
# nothing and the defaults apply.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """All settings in one read-only object."""

    # Connection details for the Weaviate Cloud Sandbox cluster
    weaviate_url: str = os.getenv("WEAVIATE_URL", "")
    weaviate_api_key: str = os.getenv("WEAVIATE_API_KEY", "")

    # Google Gemini, which writes the final answers
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

    # Name of the collection (like a table) inside Weaviate
    collection_name: str = os.getenv("COLLECTION_NAME", "DocumentKnowledge")

    # The small model that turns text into number lists (embeddings).
    # It downloads once (about 90 MB) and then runs on the local machine.
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Where the Phoenix monitoring dashboard listens
    phoenix_endpoint: str = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    )

    # How many passages to fetch for each question
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Balance between the two search methods in hybrid search:
    # 0.0 = keyword matching only, 1.0 = meaning (vector) search only,
    # 0.5 = equal weight to both.
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))


settings = Settings()


def require_settings(*names: str) -> None:
    """Stops with a clear message if a needed setting is empty.

    Example: require_settings("weaviate_url", "weaviate_api_key")
    """
    missing = [n for n in names if not getattr(settings, n)]
    if missing:
        pretty = ", ".join(n.upper() for n in missing)
        raise SystemExit(
            f"Missing settings: {pretty}. Open the .env file in the project "
            "folder and fill in these values. If you do not have a .env file "
            "yet, copy .env.example to .env first."
        )
