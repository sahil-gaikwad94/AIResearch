"""kestrel5 -- the Kestrel-5 reference implementation.

Everything public is a name from ``model.py``; importing this package never
prints anything and never runs the 15 self-assertions (they live under
``if __name__ == "__main__"`` in ``model.py``).
"""

from .model import *  # noqa: F401,F403 re-export the model namespace
from .model import (FULL, KESTREL5_NANO_165M, Kestrel5, tiny_config)
