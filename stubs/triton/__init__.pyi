# Precise local stubs for the parts of `triton` this project uses (vesper/kernels/*_triton.py).
#
# Triton ships no type information. These stubs declare only what our kernels call, with exact
# types; anything else is undeclared on purpose. A kernel's device signature (pointers as
# `language.tensor`, `constexpr` blocks) is not its launch signature: the host passes torch
# tensors for pointers and plain numbers for scalars, which `Launch` states.
from collections.abc import Callable

from torch import Tensor

class Launch:
    def __call__(self, *args: Tensor | float) -> None: ...

class JITFunction[**P]:
    def __getitem__(self, grid: tuple[int, ...]) -> Launch: ...

def jit[**P](fn: Callable[P, None]) -> JITFunction[P]: ...
def next_power_of_2(n: int) -> int: ...
def cdiv(x: int, y: int) -> int: ...
