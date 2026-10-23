"""Typed refusals: preserve API messages/statuses, expose only fixed categories."""


class TrainingAlreadyRunning(ValueError):
    usage_error_code = 'busy'
    usage_error_reason = 'training_already_running'


class CaptionBackendUnavailable(RuntimeError):
    usage_error_code = 'unavailable'
    usage_error_reason = 'caption_backend_unavailable'


class BankInputError(ValueError):
    usage_error_code = 'invalid_input'

    def __init__(self, message, reason):
        super().__init__(message)
        self.usage_error_reason = reason
