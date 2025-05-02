"""
All basic sensor objects are defined in this file, but no sensor functors,
or helper functions appear here. If you want to include sensors in your
simulation, include the file sensor.h instead.

File: sensorbase.py
Author: David Gordon
"""
# Built-in
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import numpy as np
# Third-party
from numpy import pi, rad2deg, cos, sin
from OpenGL.GL import *
# Beast
from pybeast.core.utils.vector2D import Vector2D
from pybeast.core.world.worldobject import WorldObject
# Beast Typing
if TYPE_CHECKING:
    from pybeast.core.agents.animat import Animat


class SensorMatchFunction(ABC):
    """
    Likely contains logic to determine whether a given world object matches certain criterai
    """
    def __int__(self):
        pass

    @abstractmethod
    def Reset(self):
        pass

    @abstractmethod
    def __call__(self, obj: WorldObject):
        pass

class SensorEvalFunction(ABC):
    """
    Abstract base class for evaluation functors.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, obj: WorldObject, nearest_point: Vector2D):
        """
        Must be implemented by subclasses.
        """

    @abstractmethod
    def Reset(self):
        """
        Resets the evaluation function.
        """
        pass

    @abstractmethod
    def GetOutput(self) -> float:
        """
        Must be implemented by subclasses.
        """
        pass

class SensorScaleFunction(ABC):
    """
    Abstract base class for scaling functors.
    """
    def __init__(self):
        pass

    def __call__(self, val: float) -> float:
        """
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the scaling function.
        """
        pass

SENSOR_ALPHA = 0.2  # Transparency value for Sensors

class Sensor(WorldObject):
    '''
    The Sensor class is the base class for all the different types of sensor:
    TouchSensor, SelfSensor, AreaSensor and BeamSensor. The basic Sensor class
    imposes no area restriction and so will allow its owner to detect any
    object in the world.
    '''

    def __init__(self,
                 startLocation: Optional[Vector2D] = None,
                 startOrientation: Optional[float] = None,
                 relLocation: Optional[Vector2D] = None,
                 relOrientation: float = 0.0,
                 MatchFunc: Optional[SensorMatchFunction] = None,
                 EvalFunc: Optional[SensorEvalFunction] = None,
                 ScaleFunc: Optional[SensorScaleFunction] = None
                 ):

        super().__init__(startLocation, startOrientation)

        assert -np.pi <= relOrientation <= np.pi

        self.relLocation = relLocation
        self.relOrientation = relOrientation
        self.MatchFunc = MatchFunc
        self.EvalFunc = EvalFunc
        self.ScaleFunc = ScaleFunc

        self.myOwner = None

    # -----------------------------------------------------------------------------------------------------------------
    # Interface methods of sensor class
    # -----------------------------------------------------------------------------------------------------------------

    def Init(self):
        """
        Initializes the sensor.
        """
        if self.relLocation is None:
            self.relLocation = Vector2D()

        if self.myOwner is not None:
            self.startLocation = self.myOwner.GetLocation() + self.relLocation
            orientation = self.relOrientation + self.myOwner.GetOrientation()
            if orientation < 0.0:
                orientation += 2.0*np.pi
            self.startOrientation = orientation

        super().Init()

    def Update(self):
        """
        Resets the sensor ready for the next round of tests.
        """
        self.EvalFunc.Reset()

        if self.myOwner is not None:
            newLocation = self.relLocation.Rotation(self.myOwner.GetOrientation()) + self.myOwner.GetLocation()
            self.SetLocation(newLocation)
            newOrientation = self.relOrientation + self.myOwner.GetOrientation()
            if newOrientation < 0.0:
                newOrientation += 2.0*np.pi
            self.SetOrientation(newOrientation)

    def Interact(self, other: WorldObject):
        """
        Calls the sensor's matching function on the WorldObject, then if it's a match, calls the evaluation function.
        """
        if self.MatchFunc(other):
            self.EvalFunc(other, other.GetLocation())

    def Display(self):
        """
        Displays the sensor.
        """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------------------------------------------------------

    def GetOutput(self) -> float:
        """
        Returns the sensor's output for this round. The scaling function is applied
        to the output of the evaluation function to get the result, thus the output
        might be divided by the range to yield a value [0:1], or randomly adjusted
        to simulate noise.
        """
        return self.ScaleFunc(self.EvalFunc.GetOutput())

    def GetOwner(self):
        """
        Gets the owner of the sensor.
        """
        return self.myOwner

    # -----------------------------------------------------------------------------------------------------------------
    # Mutators
    # -----------------------------------------------------------------------------------------------------------------



    def SetOwner(self, owner):
        """
        Sets the owner of the sensor.
        """
        self.myOwner = owner

    def SetMatchingFunction(self, func: SensorMatchFunction):
        """
        Sets the matching function of the sensor.
        """
        self.MatchFunc = func

    def SetEvaluationFunction(self, func: SensorEvalFunction):
        """
        Sets the evaluation function of the sensor.
        """
        self.EvalFunc = func

    def SetScalingFunction(self, func: SensorScaleFunction):
        """
        Sets the scaling function of the sensor.
        """
        self.ScaleFunc = func

class SelfSensor(Sensor):
    """
    The SelfSensor is used to detect information about its owner. An Animat can
    use a SelfSensor to get information on its location and the state of its
    controls.
    """

    def __init__(self, Type: str = 'X', ctrl: str = ""):

        super().__init__()

        assert Type in ['X', 'Y', 'Angle', 'Control'], f"Sensor type 'Type' must be in {['X', 'Y', 'Angle', 'Control']}"
        self.myType = Type
        self.controlName = ctrl  # From Animat controls, e.g., "left"

    def GetOutput(self) -> float:
        """
        Returns the SelfSensor's output, which comes directly from the
        sensor's owner. Currently supports sensing of x/y location,
        angle, or control output.
        """
        if self.myOwner is None:
            return 0.0

        if self.myType == 'X':
            return self.myOwner.GetLocation().x / self.myOwner.GetWorld().GetWidth()
        elif self.myType == 'Y':
            return self.myOwner.GetLocation().y / self.myOwner.GetWorld().GetHeight()
        elif self.myType == 'Angle':
            return self.myOwner.GetOrientation() / (2 * pi)
        elif self.myType == 'Control':
            return self.myOwner.GetControls().get(self.controlName, 0.0)

        return 0.0

class AreaSensor(Sensor):
    """
    Detects objects within an area specified by the size and shape of the
    AreaSensor. Currently only detects objects when their location is within
    the area (i.e., just touching the area won't trip the sensor).
    """

    def Interact(self, other: WorldObject):
        """
        Checks if the WorldObject is the correct type using MatchFunc, then checks
        if the object's centre is inside the AreaSensor and calls the EvalFunc.
        """
        vec, _ = other.GetNearestPoint(self)

        if self.MatchFunc and self.EvalFunc and self.IsInside(vec):
            self.EvalFunc(other, vec)

class TouchSensor(Sensor):
    """
    Detects objects which are touching the sensor's owner.
    """

    def Init(self):
        """
        Initializes sensor radius to match that of owner.
        """
        if self.myOwner:
            self.SetRadius(self.myOwner.GetRadius())

        super().Init()
    def Interact(self, other: WorldObject):
        """
        Checks if the WorldObject is the correct type using MatchFunc, then checks
        if the sensor's owner is touching the WorldObject and calls the EvalFunc.
        """
        if self.MatchFunc and self.myOwner and self.myOwner.IsTouching(other):
            self.EvalFunc(other, other.GetNearestPoint(self))

BEAM_DRAW_QUALITY = 0.1

class BeamSensor(Sensor):
    """
    BeamSensors can really be three distinct kinds of sensor:
    - Lasers, which just detect objects a certain distance away in a straight
    line from the sensor's origin.
    - Scoped sensors, which detect objects within a certain range and a
    specified angle.
    - Unidirectional sensors which detect objects a certain distance away at
    any angle
    The three types of sensor are achieved by specifying scopes of 0, [0 < TWOPI]
    and TWOPI respectively.
    Note that BeamSensors are the most computationally expensive sensors, so if
    you can substitute them for another kind of Sensor, do so.
    """
    def __init__(self,
        scope: float = pi / 4,
        range: float = 250.0,
        location: Optional[Vector2D] = None,
        orientation: float = 0.0,
        relLocation: Optional[Vector2D] = None,
        relOrientation: float = 0,
        MatchFunc: SensorMatchFunction = None,
        EvalFunc: SensorEvalFunction = None,
        ScaleFunc: SensorScaleFunction = None,
        wrap: bool = False,
        drawFixed: bool = True,
        drawScale: float = 1.0,
        beamDrawQuality: float = 0.1):
        """
         Initializes a BeamSensor object.

         :param scope: The width of the beam in radians.
         :param range: Sets the maximum distance.
         :param location: The location of the sensor.
         :param orientation: The orientation of the sensor.
         :param drawScale: Scaling factor used in Display
         :param drawFixed: Whether to scale display according to output
         :param wrapping: The orientation of the sensor.
         :param beamDrawQuality: Draw quality for BeamSensor
         """

        super().__init__(location, orientation, relLocation, relOrientation, MatchFunc, EvalFunc, ScaleFunc)

        assert 0.0 <= scope <= 2*pi, "scope 's' must be larger equal than 0 and smaller equal than two pi"

        self.scope = scope
        self.range = range

        self.drawScale = drawScale
        self.drawFixed = drawFixed
        self.wrap = wrap
        self.wrapping = {'Left': False, 'Right': False, 'Bottom': False, 'Top': False}
        self.beamDrawQuality = beamDrawQuality

    def __repr__(self):
        return self._repr(scope=self.scope, range=self.range, owner=self.myOwner)

    # -----------------------------------------------------------------------------------------------------------------
    # Class interface
    # -----------------------------------------------------------------------------------------------------------------

    def Update(self):
        """
        Checks if the sensor is wrapping, then sets wrap locations accordingly.
        """
        if self.wrap:
            self.wrapping['Left'] = self.GetLocation().x - self.range < 0
            self.wrapping['Bottom'] = self.GetLocation().y - self.range < 0
            self.wrapping['Right'] = self.GetLocation().x + self.range > self.myOwner.GetWorld().GetWidth()
            self.wrapping['Top'] = self.GetLocation().y + self.range > self.myOwner.GetWorld().GetHeight()

        # This update sensor position and orientation
        super().Update()

    def Display(self):
        """
        Displays the BeamSensor.

        If the sensor is not visible, the display is skipped.
        If wrapping is enabled, the display is duplicated around the edges of the world.
        """
        self._Display()

        if not self.wrap:
            return

        if self.wrapping['Left']:
            temp = self.GetLocation().x
            self.SetLocationX(temp + self.myOwner.GetWorld().GetWidth())
            self._Display()
            self.SetLocationX(temp)

        if self.wrapping['Bottom']:
            temp = self.GetLocation().y
            self.SetLocationY(temp + self.myOwner.GetWorld().GetHeight())
            self._Display()
            self.SetLocationY(temp)

        if self.wrapping['Right']:
            temp = self.GetLocation().x
            self.SetLocationX(temp - self.myOwner.GetWorld().GetWidth())
            self._Display()
            self.SetLocationX(temp)

        if self.wrapping['Top']:
            temp = self.GetLocation().y
            self.SetLocationY(temp - self.myOwner.GetWorld().GetHeight())
            self._Display()
            self.SetLocationY(temp)

    def _Display(self):
        """
        Positions the matrix according to the location of the ownerAnimat and
        draws the sensor's display list.
        """
        glPushMatrix()
        glTranslated(self.GetLocation().x, self.GetLocation().y, 0)
        glRotated(rad2deg(self.GetOrientation()), 0.0, 0.0, 1.0)

        if self.drawFixed:
            scale = self.drawScale
        # Only draw sensor up to object boundary, only workds if output is normalized distance
        else:
            scale = self.drawScale - self.GetOutput()

        glScaled(scale, scale, 1.0)
        glCallList(self.GetDisplayList())
        glPopMatrix()

    def Draw(self):
        """
        Draws an alpha-blended line, segment, or circle depending on the scope
        of the sensor. The number of points in the circle is determined by the
        scope and range of the sensor.
        """
        glEnable(GL_BLEND)

        if self.scope == 0.0:
            glBegin(GL_LINES)
            glLineWidth(1.0)
            glColor4f(self.myOwner.GetColour(0), self.myOwner.GetColour(1), self.myOwner.GetColour(2), SENSOR_ALPHA)
            glVertex2d(0.0, 0.0)
            glColor4f(self.myOwner.GetColour(0), self.myOwner.GetColour(1), self.myOwner.GetColour(2), SENSOR_ALPHA * 2.0)
            glVertex2d(self.range, 0.0)
            glEnd()
        else:
            numArcPoints = int(self.scope * self.range * BEAM_DRAW_QUALITY)
            angles = np.linspace(-0.5*self.scope, 0.5*self.scope, numArcPoints, endpoint=True)
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(self.GetColour()[0], self.GetColour()[1], self.GetColour()[2], 0.0)
            glVertex2d(0.0, 0.0)
            glColor4f(self.GetColour()[0], self.GetColour()[1], self.GetColour()[2], SENSOR_ALPHA)
            for angle in angles:
                glVertex2d(self.range * cos(angle), self.range * sin(angle))
            glEnd()

        glDisable(GL_BLEND)

        return

    def InScope(self, vec: Vector2D):
        """
        Checks to see if a point is within the current testing angle of the sensor.
        """

        if self.scope == 2*pi:
            return True

        angleToOther = (vec - self.GetLocation()).GetAngle()

        startAngle = self.GetOrientation() - 0.5*self.scope
        if startAngle < 0:
            startAngle += 2*np.pi

        endAngle = self.GetOrientation() + 0.5*self.scope
        endAngle = endAngle % (2.0 * np.pi)

        # If the scope does not cross 0 degrees
        if startAngle < endAngle:
            return startAngle <= angleToOther <= endAngle
        # If the scope crosses 0 degrees
        else:
            return startAngle <= angleToOther or angleToOther <= endAngle


    # ------------------------------------------------------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------------------------------------------------------
    def GetScope(self) -> float:
        return self.scope

    def GetRange(self) -> float:
        return self.range

    # ------------------------------------------------------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------------------------------------------------------

    def SetDrawScale(self, d: float):
        self.drawScale = d

    def SetDrawFixed(self, f: bool):
        self.drawFixed = f

    def SetWrapping(self, w: bool):
        self.wrap = w


class SignalSensor(Sensor):
    """A sensor that detects signals from other animats."""
    
    def __init__(self):
        super().__init__()
        self.signal_value = 0.0
        self.signal_angle = 0.0
        self.has_signal = 0.0
    
    def Interact(self, other):
        """
        Override the base Interact method - signal sensors don't use the standard
        matching/evaluation functions as they process signals differently.
        """
        # Signal sensors don't use the interact method for processing signals
        # They get signals directly through the animat's received_signals
        pass
    
    def Update(self):
        """Update the sensor based on received signals."""
        self.signal_value = 0.0
        self.signal_angle = 0.0
        self.has_signal = 0.0
        
        animat = self.GetOwner()
        if not animat:
            return
            
        # Process received signals
        signals = animat.GetReceivedSignals()
        if signals:
            strongest_signal = 0.0
            for k, v in signals.items():
                if v["strength"] > strongest_signal:
                    strongest_signal = v["strength"]
                    self.signal_value = v["value"]
                    
                    if "angle" in v:
                        # Normalize angle to range [-1, 1] for the neural network
                        angle = v["angle"]
                        # Ensure angle is in [-pi, pi] range
                        while angle > np.pi:
                            angle -= 2 * np.pi
                        while angle < -np.pi:
                            angle += 2 * np.pi
                        self.signal_angle = angle / np.pi
            
            if strongest_signal > 0:
                self.has_signal = 1.0
    
    def GetOutput(self):
        """Return the sensor outputs as a list for the neural network."""
        # return [self.has_signal, self.signal_value, self.signal_angle]
        return [self.signal_value, self.signal_angle]
    
    def Display(self):
        """Optionally display the sensor (can be empty)."""
        pass