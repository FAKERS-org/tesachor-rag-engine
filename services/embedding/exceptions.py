class EmbeddingProviderError(Exception):
    """Base exception for embedding providers"""
    pass

class ProviderNotConfiguredError(EmbeddingProviderError):
    """Raised when provider is missing required config"""
    pass

class EmbeddingAPIError(EmbeddingProviderError):
    """Raised when API call fails"""
    pass