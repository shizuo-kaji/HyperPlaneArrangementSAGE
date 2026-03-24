"""Hyperplane arrangement utilities for Sage."""
from . import arrangement as _arrangement
from . import tangential_field as _tfield
from . import fit as _fit

__all__ = []
for module in (_arrangement, _tfield, _fit):
    names = getattr(module, '__all__', [])
    __all__.extend(names)
    globals().update({name: getattr(module, name) for name in names})
