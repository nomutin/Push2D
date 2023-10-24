"""Type stubs for Push2D."""

from __future__ import annotations

from nptyping import Int, NDArray, Shape

# up, down, left, right
Act = NDArray[Shape["4"], Int]
Obs = NDArray[Shape["Width, Height, 3"], Int]
