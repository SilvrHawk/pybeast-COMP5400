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

        self.vocab_size = 1

    def normalize_value(self, value, vocab_size):
        """
        Normalize a value from range [1, vocab_size] to [0, 1]
        """
        if vocab_size <= 1:
            return 0.0

        return (value - 1) / (vocab_size - 1)

    def get_color_for_value(self, value):
        """
        Generate a color based on signal value in the range [-1, 1].
        Uses a continuous colormap that transitions through:
        Red > Orange > Yellow > Green > Cyan > Blue > Purple

        If vocab_size is 1, always returns red color.
        """
        if self.vocab_size == 1:
            return [1.0, 0.0, 0.0, 0.3]

        value = max(0.0, min(1.0, value))

        colors = [
            (0.0, [1.0, 0.0, 0.0, 0.3]),  # Red
            (0.2, [1.0, 0.5, 0.0, 0.3]),  # Orange
            (0.4, [1.0, 1.0, 0.0, 0.3]),  # Yellow
            (0.6, [0.0, 1.0, 0.0, 0.3]),  # Green
            (0.8, [0.0, 1.0, 1.0, 0.3]),  # Cyan
            (1.0, [0.0, 0.0, 1.0, 0.3]),  # Blue
        ]

        # Find the two color points to interpolate between
        for i in range(len(colors) - 1):
            pos1, color1 = colors[i]
            pos2, color2 = colors[i + 1]

            if value <= pos2:
                # Calculate interpolation factor
                t = (value - pos1) / (pos2 - pos1) if (pos2 - pos1) > 0 else 0

                # Interpolate between colors
                result = [
                    color1[0] + t * (color2[0] - color1[0]),
                    color1[1] + t * (color2[1] - color1[1]),
                    color1[2] + t * (color2[2] - color1[2]),
                    0.3,
                ]
                return result

        return [1.0, 0.0, 0.0, 0.3]

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
            # Use color based on normalized signal value
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

    def Activate(
        self, location: Vector2D, strength: float, value: float, vocab_size=1.0
    ):
        """
        Start displaying the signal at the given location.

        Args:
            location: Position to display the signal
            strength: Radius/intensity of the signal
            value: Signal value (integer in range [1, vocab_size])
            vocab_size: Size of the vocabulary (optional)
        """
        self.active = True
        self.location = location
        self.strength = strength
        self.vocab_size = vocab_size

        normalized_value = self.normalize_value(value, self.vocab_size)
        self.value = normalized_value
        self.activationTime = time.time()

    def Deactivate(self):
        """
        Stop displaying the signal.
        """
        self.active = False

    def Update(self, location: Vector2D = None, value: float = None):
        """
        Update the signal position.
        """
        if location:
            self.location = location
        if value:
            normalized_value = self.normalize_value(value, self.vocab_size)
            self.value = normalized_value

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
