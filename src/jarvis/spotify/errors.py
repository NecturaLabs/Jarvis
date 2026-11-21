"""Spotify failures, told to the model as an operation's failure.

Words that Spotify wrote (error messages, names) are untrusted: an error built from them carries
`origin=Origin.SPOTIFY`; Jarvis's own fixed sentences keep the default trusted origin.
"""

from jarvis.operations import OperationError


class SpotifyError(OperationError):
    """Spotify refused or failed a request."""


class NoDeviceError(SpotifyError):
    """No Spotify player is available to play on."""


class SpotifyAuthError(SpotifyError):
    """Login or token refresh failed."""


class SpotifyLoginExpired(  # ruff: ignore[error-suffix-on-exception-name] - the name callers use
    SpotifyAuthError
):
    """Spotify no longer accepts the stored login; the owner must log in again."""
