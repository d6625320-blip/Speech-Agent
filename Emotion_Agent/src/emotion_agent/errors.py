class AgentError(RuntimeError):
    """An actionable error safe to display without provider response bodies."""


class ConfigurationError(AgentError):
    pass


class ProviderError(AgentError):
    pass


class AudioError(AgentError):
    pass
