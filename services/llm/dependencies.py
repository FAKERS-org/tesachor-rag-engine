from config import config
from providers.openrouter import OpenRouterLLMProvider
from exceptions import ProviderNotConfiguredError

# add providers here
PROVIDERS = {
    "openrouter": OpenRouterLLMProvider,
}

def get_provider():
    provider_class = PROVIDERS.get(config.provider)
    
    # if not found
    if not provider_class:
        raise ProviderNotConfiguredError(f"Provider {config.provider} is not supported or config")
    
    # init provider config
    return provider_class(config=config)
