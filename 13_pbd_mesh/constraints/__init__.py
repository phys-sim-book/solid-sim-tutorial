"""
Constraints Package

This package contains all constraint solvers for the XPBD mesh simulation:
- edge: Edge length constraints for structural stability
- volume: Volume constraints for tetrahedra to prevent collapse
"""

from .edge import solve_edges
from .volume import solve_volumes

__all__ = [
    'solve_edges',
    'solve_volumes',
]
