"""Type stubs for Push2D."""

from nptyping import Int, NDArray, Shape

# up, down, left, right
Act = NDArray[Shape["4"], Int]
Obs = NDArray[Shape["Width, Height, 3"], Int]
