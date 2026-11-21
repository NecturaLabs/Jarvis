# Minimal stub for the parts of onnxruntime Jarvis uses (the package ships no type hints).
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

class SessionOptions:
    inter_op_num_threads: int
    intra_op_num_threads: int
    def __init__(self) -> None: ...
    def add_session_config_entry(self, key: str, value: str) -> None: ...
    def get_session_config_entry(self, key: str) -> str: ...

class NodeArg:
    name: str
    shape: Sequence[int | str | None]
    type: str

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str,
        sess_options: SessionOptions | None = ...,
        providers: Sequence[str] | None = ...,
    ) -> None: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_modelmeta(self) -> ModelMetadata: ...
    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[np.float32]]: ...

class ModelMetadata:
    custom_metadata_map: dict[str, str]
