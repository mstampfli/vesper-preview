"""Vesper: a language model that thinks in time instead of depth.

Importing the package installs beartype on every module below it, so the tensor contracts in
docs/registry.md are checked on real calls, not only by the static checkers.
"""

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# the Triton kernels are compiled for the GPU by ``triton.jit``, never called as Python
# functions, so there is nothing for beartype to check there (and it cannot wrap them)
beartype_this_package(
    conf=BeartypeConf(
        claw_skip_package_names=(
            "vesper.kernels.nll_triton",
            "vesper.kernels.norm_triton",
            "vesper.kernels.keys_triton",
            "vesper.kernels.act_triton",
        )
    )
)
