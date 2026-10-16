"""Vesper: a language model that thinks in time instead of depth.

Importing the package installs beartype on every module below it, so the tensor contracts in
docs/registry.md are checked on real calls, not only by the static checkers.
"""

from beartype.claw import beartype_this_package

beartype_this_package()
