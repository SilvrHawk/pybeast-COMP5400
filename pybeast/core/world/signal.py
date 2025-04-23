# Built-in
import time

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

    def Display(self):
        """Render the signal as a solid disk."""
        # Don't display anything if radius is 0 or very small
        if self.strength < 1.0:
            return

        # Save current position for drawing
        glPushMatrix()
        glTranslated(self.location.GetX(), self.location.GetY(), 0.0)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Determine color and alpha based on state
        if self.active:
            # Red solid disk when active
            displayColor = [1.0, 0.2, 0.2, 0.25]  # Red with transparency
        else:
            # Cyan solid disk when inactive
            displayColor = [0.2, 0.8, 1.0, 0.08]  # Faint cyan

        radius = self.strength

        # Draw the solid disk
        glColor4f(displayColor[0], displayColor[1], displayColor[2], displayColor[3])
        signalDisk = gluNewQuadric()
        gluQuadricDrawStyle(signalDisk, GLU_FILL)
        gluDisk(signalDisk, 0.0, radius, 32, 1)  # Start radius from 0 for solid disk
        gluDeleteQuadric(signalDisk)

        glDisable(GL_BLEND)
        glPopMatrix()

    def Activate(self, location: Vector2D, strength: float, value: float):
        """
        Start displaying the signal at the given location.
        """
        self.active = True
        self.location = location
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
