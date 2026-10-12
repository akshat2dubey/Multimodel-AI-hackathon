#!/usr/bin/env python3
"""
HTTPS-only URL opening (T36).

``urllib.request.urlopen`` accepts ``file:``, ``ftp:`` and custom schemes, which
is why a static analyser flags every call site. Every URL this project fetches
is an HTTPS API or bucket, so the scheme is asserted once, here, and each call
site goes through this wrapper instead of reaching for urllib directly.

Usage::

    from net_safe import urlopen
    with urlopen(request, timeout=120) as response:
        data = response.read()
"""

from __future__ import annotations

import urllib.parse
import urllib.request

ALLOWED_SCHEMES = ("https",)


def checked_url(url: str) -> str:
    """Return the URL unchanged, or raise if it is not HTTPS."""
    scheme = urllib.parse.urlparse(str(url)).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"refusing to fetch a non-HTTPS URL (scheme={scheme or 'none'}); "
            f"allowed: {', '.join(ALLOWED_SCHEMES)}"
        )
    return url


def urlopen(request, timeout: float = 60):
    """``urllib.request.urlopen`` with an HTTPS scheme check in front of it.

    Accepts a Request or a plain URL string. Raises ValueError (not a network
    error) for a disallowed scheme, so a caller cannot mistake it for a retry.
    """
    url = getattr(request, "full_url", request)
    checked_url(url)
    # The scheme was asserted by checked_url() above.
    return urllib.request.urlopen(request, timeout=timeout)  # nosec B310
