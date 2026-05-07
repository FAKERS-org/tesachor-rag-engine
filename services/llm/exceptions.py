class LLMError(Exception):
    """Base exception for LLM service"""
    pass

class ProviderNotConfiguredError(LLMError):
    """Raised when a provider is selected but missing configuration"""
    pass

class LLMAPIError(LLMError):
    """Raised when the provider API returns an error"""
    pass