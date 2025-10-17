from importlib import metadata

from langchain_parallel_web.chat_models import ChatParallelWeb
from langchain_parallel_web.extract_tool import ParallelExtractTool
from langchain_parallel_web.search_tool import ParallelWebSearchTool

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatParallelWeb",
    "ParallelExtractTool",
    "ParallelWebSearchTool",
    "__version__",
]
