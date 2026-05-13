import sys
from unittest.mock import MagicMock

# Mock heavy dependencies so tests don't require the full llama_index/torch stack
_mocks = [
    "llama_index",
    "llama_index.llms",
    "llama_index.llms.gemini",
    "llama_index.core",
]
for mod in _mocks:
    sys.modules.setdefault(mod, MagicMock())
