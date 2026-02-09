# gemini offers text embedding API

from portable_brain.common.services.embedding_service.text_embedding.dispatcher import TypedTextEmbeddingClient, TextEmbeddingProvider
from portable_brain.common.services.embedding_service.text_embedding.protocols import TypedTextEmbeddingProtocol

# NOTE: only supports the generic wrappers here
__all__ = ["TypedTextEmbeddingClient", "TextEmbeddingProvider", "TypedTextEmbeddingProtocol"]