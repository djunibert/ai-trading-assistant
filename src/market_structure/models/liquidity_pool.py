"""
Objet métier : LiquidityPool.

Représente une zone de liquidité du marché.
"""

from dataclasses import dataclass


@dataclass
class LiquidityPool:
    direction: str
    price: float
    created_index: int
    strength: int = 1
    age: int = 0
    swept: bool = False
    active: bool = True
    score: float = 0.0