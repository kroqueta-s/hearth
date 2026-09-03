# SPDX-License-Identifier: MIT
"""The stdout guard. **Copy this file as it is.**

This is the smallest piece of the contract and the easiest to skip, and skipping
it produces the worst kind of bug: a generation that works perfectly and then
fails to be read, because a library printed one line to stdout in the middle of
it.

Model code prints. It prints version banners, deprecation notices, and progress
that its author wanted a human to see. **Replacing `sys.stdout` is not enough**,
because a compiled extension writes to file descriptor 1 directly and never
looks at `sys.stdout` at all.

So the real stdout is duplicated to a new descriptor, which becomes the
protocol's private channel, and `sys.stdout` is pointed at stderr. Anything that
prints now lands in stderr, where it belongs and where nobody parses it.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


def install_stdout_guard() -> TextIO:
    """Duplicate the real stdout, hide it, and point `sys.stdout` at stderr.

    **Call this before importing anything that might print**, which in practice
    means before importing the model at all.

    Returns:
        The protocol's stream. **Nothing else may write to it.**
    """
    fd = os.dup(sys.stdout.fileno())
    protocol = os.fdopen(fd, "w", encoding="utf-8", newline="\n", buffering=1)
    sys.stdout = sys.stderr
    return protocol
