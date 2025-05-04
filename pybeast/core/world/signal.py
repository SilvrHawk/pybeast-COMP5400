# Built-in
import time
import numpy as np

# Third-party
from OpenGL.GL import *
from OpenGL.GLU import (
    gluNewQuadric,
    gluQuadricDrawStyle,
    gluDisk,
    gluDeleteQuadric,
    GLU_FILL,
)

# Beast
from pybeast.core.world.drawable import Drawable
from pybeast.core.utils.vector2D import Vector2D


class Signal:
    """
    Maintains and displays signal indicators for agents.
    """

    def __init__(
        self,
        Visible: bool = True,
        r: float = 0.2,
        g: float = 0.8,
        b: float = 1.0,
        a: float = 0.3,
    ):
        self.colour = [r, g, b, a]
        self.visible = Visible

        # Signal state
        self.active = False
        self.value = 0.0
        self.strength = 50.0
        self.location = Vector2D()
        self.activationTime = 0.0

    def get_color_for_value(self, value):
        """
        Generate a color based on signal value in the range [-1, 1].
        - Negative values (-1 to 0): Blue to White gradient
        - Positive values (0 to 1): White to Red gradient
        """
        # Clamp value to [-1, 1] range
        value = max(-1.0, min(1.0, value))

        alpha = 0.3

        if value < 0:
            # For negative values: gradient from blue (-1) to cyan (closer to 0)
            intensity = abs(value)  # 0 to 1
            return [0.0, intensity, 1.0, alpha]
        elif value > 0:
            # For positive values: gradient from yellow (closer to 0) to red (1)
            intensity = value  # 0 to 1
            return [1.0, 1.0 - (intensity * 0.8), 0.0, alpha]
        else:
            # Exactly zero: neutral green
            return [0.0, 1.0, 0.0, alpha]

    def Display(self):
        """Render the signal as a solid disk with color based on value."""
        # Don't display anything if radius is 0 or very small
        if self.strength < 1.0:
            return

        # Save current position for drawing
        glPushMatrix()
        glTranslated(self.location.GetX(), self.location.GetY(), 0.0)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Determine color and alpha based on state and value
        if self.active:
            # Use color based on signal value in [-1, 1] range
            displayColor = self.get_color_for_value(self.value)
        else:
            # Transparent when inactive
            displayColor = [0.0, 0.0, 0.0, 0.0]

        radius = self.strength

        glColor4f(displayColor[0], displayColor[1], displayColor[2], displayColor[3])
        signalDisk = gluNewQuadric()
        gluQuadricDrawStyle(signalDisk, GLU_FILL)
        gluDisk(signalDisk, 0.0, radius, 32, 1)
        gluDeleteQuadric(signalDisk)

        glDisable(GL_BLEND)
        glPopMatrix()

    def Activate(self, location: Vector2D, strength: float, value: float):
        """
        Start displaying the signal at the given location.
        """
        self.active = True
        self.location = location
        self.strength = strength
        self.value = value
        self.activationTime = time.time()

    def Deactivate(self):
        """
        Stop displaying the signal.
        """
        self.active = False

    def Update(self, location: Vector2D = None):
        """
        Update the signal position.
        """
        if location:
            self.location = location

    def IsActive(self) -> bool:
        """
        Return whether the signal is currently active.
        """
        return self.active

    def SetColour(self, r: float, g: float, b: float, a: float = 0.3):
        """
        Set the signal color.
        """
        self.colour = [r, g, b, a]
