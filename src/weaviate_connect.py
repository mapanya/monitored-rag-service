"""Opens the connection to the Weaviate Cloud cluster.

This lives in its own file so that every part of the project connects
in exactly the same way. The waiting times are set generously on
purpose: the health check that runs when connecting gets 30 seconds
instead of the default few seconds, because the cluster can sit in a
far away data centre and a long distance connection needs more
patience.
"""

import weaviate
import weaviate.classes as wvc

from src.config import require_settings, settings


def connect() -> weaviate.WeaviateClient:
    """Connects to the cluster named in the .env file."""
    require_settings("weaviate_url", "weaviate_api_key")
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.weaviate_url,
        auth_credentials=wvc.init.Auth.api_key(settings.weaviate_api_key),
        additional_config=wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(
                init=30,     # seconds allowed for the connection check
                query=120,   # seconds allowed for a search
                insert=180,  # seconds allowed for loading documents
            )
        ),
    )
