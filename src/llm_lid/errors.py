class LLMLIDError(Exception):
    """Base class for llm-lid errors."""


class ResponseError(LLMLIDError):
    """The model's reply could not be parsed into one valid label per token."""


class LabelingError(LLMLIDError):
    """The model failed to return valid labels after all retries."""

    def __init__(self, message, raw_responses=None):
        super().__init__(message)
        self.raw_responses = raw_responses or []
