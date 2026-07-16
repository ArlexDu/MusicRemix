"""MusicRemix: 歌曲换音色工具。

保留原曲旋律、歌词与伴奏，仅替换演唱者音色。
"""

# faiss 与 torch 共存时会因 OpenMP 重复加载导致段错误，提前放开限制
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
