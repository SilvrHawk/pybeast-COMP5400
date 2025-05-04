"""
BEASTFrame module.

This module defines the main window for the BEAST application.
"""
import copy
import pickle
import sys
# from built-in
import time
from pathlib import Path
import importlib.util
import threading
from typing import Optional, Dict

# from third-party
import wx

# from pybeast
from pybeast.core.logger import BeastLogWindow
from pybeast.core.simulation import Simulation
from pybeast.core.world.world import World
from pybeast.paths import DEMO_DIR, DATA_DIR
from pybeast.gui.worldglcanvas import WorldGLCanvas

class AppIDs:
    ID_FILE_STARTSIM0 = wx.ID_HIGHEST + 1
    ID_FILE_STARTSIM1 = ID_FILE_STARTSIM0 + 1
    ID_FILE_STARTSIM2 = ID_FILE_STARTSIM0 + 2
    ID_FILE_STARTSIM3 = ID_FILE_STARTSIM0 + 3
    ID_FILE_STARTSIM4 = ID_FILE_STARTSIM0 + 4
    ID_FILE_STARTSIM5 = ID_FILE_STARTSIM0 + 5
    ID_FILE_STARTSIM6 = ID_FILE_STARTSIM0 + 6
    ID_FILE_STARTSIM7 = ID_FILE_STARTSIM0 + 7
    ID_FILE_STARTSIM8 = ID_FILE_STARTSIM0 + 8
    ID_FILE_STARTSIM9 = ID_FILE_STARTSIM0 + 9
    ID_FILE_STARTSIM10 = ID_FILE_STARTSIM0 + 10

    ID_FILE_STARTMYSIM = ID_FILE_STARTSIM9 + 1

    ID_FILE_LOAD = ID_FILE_STARTMYSIM + 10
    # ID_FILE_LOAD0 = ID_FILE_LOAD + 1
    # ID_FILE_LOAD1 = ID_FILE_LOAD + 2
    # ID_FILE_LOAD2 = ID_FILE_LOAD + 3
    # ID_FILE_LOAD3 = ID_FILE_LOAD + 4
    # ID_FILE_LOAD4 = ID_FILE_LOAD + 5
    # ID_FILE_LOAD5 = ID_FILE_LOAD + 6
    # ID_FILE_LOAD6 = ID_FILE_LOAD + 7
    # ID_FILE_LOAD7 = ID_FILE_LOAD + 8
    # ID_FILE_LOAD8 = ID_FILE_LOAD + 9
    # ID_FILE_LOAD9 = ID_FILE_LOAD + 10

    ID_FILE_SAVE = ID_FILE_LOAD + 1
    # ID_FILE_SAVE0 = ID_FILE_SAVE + 1
    # ID_FILE_SAVE1 = ID_FILE_SAVE + 2
    # ID_FILE_SAVE2 = ID_FILE_SAVE + 3
    # ID_FILE_SAVE3 = ID_FILE_SAVE + 4
    # ID_FILE_SAVE4 = ID_FILE_SAVE + 5
    # ID_FILE_SAVE5 = ID_FILE_SAVE + 6
    # ID_FILE_SAVE6 = ID_FILE_SAVE + 7
    # ID_FILE_SAVE7 = ID_FILE_SAVE + 8
    # ID_FILE_SAVE8 = ID_FILE_SAVE + 9
    # ID_FILE_SAVE9 = ID_FILE_SAVE + 10

    ID_DISP_ANIMATS = wx.ID_HIGHEST + 51
    ID_DISP_OBJECTS = ID_DISP_ANIMATS + 1
    ID_DISP_TRAILS = ID_DISP_ANIMATS + 2
    ID_DISP_COLLISIONS = ID_DISP_ANIMATS + 3
    ID_DISP_SENSORS = ID_DISP_ANIMATS + 4
    ID_DISP_MONITOR = ID_DISP_ANIMATS + 5

    ID_SIM_PAUSE = wx.ID_HIGHEST + 61
    ID_SIM_RESUME = ID_SIM_PAUSE + 1
    ID_SIM_RESET = ID_SIM_PAUSE + 2
    ID_SIM_FAST = ID_SIM_PAUSE + 3
    ID_SIM_SAVE = ID_SIM_PAUSE + 4

    ID_WORLD_NEXT = wx.ID_HIGHEST + 71
    ID_WORLD_PREV = ID_WORLD_NEXT + 1
    ID_WORLD_3D = ID_WORLD_NEXT + 2
    ID_WORLD_2D = ID_WORLD_NEXT + 3

    ID_DEMOS = wx.ID_HIGHEST + 1
    ID_HELP_ABOUT = wx.ID_HIGHEST + 81
    ID_MAIN_TIMER = wx.ID_HIGHEST + 101

class BeastFrame(wx.Frame):
    """
    BeastFrame class.

    The main window for the BEAST application.

    """
    demosDir = Path('../')


    def __init__(
            self,
            frame: wx.Frame,
            title: str,
            pos: wx.Point = None,
            size: wx.Size = None,
            style: int = wx.DEFAULT_FRAME_STYLE,
            includeDemos: bool = True,
            mySimulation: Optional[Simulation] = None
    ):
        """
        Initialize BeastFrame.

        Args:
            frame (wx.Frame): The parent frame.
            title (str): The title of the frame.
            pos (wx.Point): The position of the frame.
            size (wx.Size): The size of the frame.
            plugin (str): The plugin to be loaded.
            style (long): The style of the frame.

        """
        if pos is None:
            pos = wx.Point(50, 50)
        if size is None:
            size = wx.Size(808, 681)

        super().__init__(frame, -1, title, pos, size, style)

        self.mySimulation = mySimulation
        self.simulationNames = []
        self.simulationClass = []
        self.currentSimulation = None
        self.worldCanvas = None
        self.currentThread = None
        self.menuBar = None
        self.dmosBar = None
        self.statusBar = None
        self.logWindow = None
        self.currentSimId = -1
        self.fps = 33
        self.started = False
        self.paused = False
        self.initialSimulationCopy = None
        self.includeDemos = includeDemos

        self.CreateMenuBar()
        self.CreateLogWindow()

        self.statusBar = self.CreateStatusBar(2)
        self.statusBar.SetStatusText("Ready", 0)

        self.BindEvents()
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def LoadDemos(self):
        """
        Loads file from demo directory if IsDemo attribute is set to True
        """

        # Iterate over all *.py files in demo directory
        for demoPath in [p for p in DEMO_DIR.iterdir() if p.suffix == '.py']:

            # Load module dynamically
            moduleName = demoPath.stem
            spec = importlib.util.spec_from_file_location(moduleName, demoPath)
            demoModule = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demoModule)

            # If module's IsDemo attribute is set to True add Simulation Class
            if hasattr(demoModule, 'IsDemo'):
                if demoModule.IsDemo:
                    sys.modules[moduleName] = demoModule
                    assert hasattr(demoModule, 'GUIName'), "demo modules must have string attribute 'GUIName'"
                    assert hasattr(demoModule, 'SimClassName'), ("demo modules must a string attribute 'SimClassName',"
                        "so that BEAST knows which Simulation class to load")

                    GuiName = getattr(demoModule, 'GUIName')
                    SimClassName = getattr(demoModule, 'SimClassName')
                    SimClass = getattr(demoModule, SimClassName)

                    self.simulationNames.append(GuiName)
                    self.simulationClass.append(SimClass)

        return

    #================================================================================================
    # Create and populate menubar (GUI user interface)
    #================================================================================================

    def CreateMenuBar(self):
        """
        Create and set up the menu bar for the BeastFrame.

        This method initializes and populates the menu bar with menus and menu items
        for different functionalities such as file operations, view options, simulation controls,
        world navigation, and help. It also associates command identifiers (IDs) with menu items
        and checks some items by default.
        """

        # Create menubar
        self.menuBar = wx.MenuBar()

        # File menu
        fileMenu = wx.Menu()
        fileMenu.Append(AppIDs.ID_FILE_SAVE, f'&Save')
        fileMenu.Append(AppIDs.ID_FILE_LOAD, f'&Load')
        self.menuBar.Append(fileMenu, "&File")

        # Demos
        if self.includeDemos:
            self.LoadDemos()
            self.demoMenu = wx.Menu()
            for i, demoName in enumerate(self.simulationNames):
                self.demoMenu.Append(getattr(AppIDs, f'ID_FILE_STARTSIM{i}'), f'&{demoName}', f"Start demo {demoName}")

            self.menuBar.Append(self.demoMenu, "&Demos")

        # Simulation menu
        simMenu = wx.Menu()

        if self.mySimulation is not None:
            simMenu.Append(AppIDs.ID_FILE_STARTMYSIM, "&Start", "Pauses the simulation")

        simMenu.Append(AppIDs.ID_SIM_PAUSE, "&Pause", "Pauses the simulation")
        simMenu.Append(AppIDs.ID_SIM_RESUME, "&Resume", "Resumes the simulation")
        simMenu.Append(AppIDs.ID_SIM_FAST, "High speed", "Turns off the display and runs at top speed")
        simMenu.Append(AppIDs.ID_SIM_RESET, "&Reset", "Clears all simulation data and restart")
        self.menuBar.Append(simMenu, "&Simulation")

        # Help menu
        helpMenu = wx.Menu()
        self.menuBar.Append(helpMenu, "&Help")
        helpMenu.Append(AppIDs.ID_HELP_ABOUT, "&About")

        # TODO: Don't think this does much
        # View menu
        # viewMenu = wx.Menu()
        # self.menuBar.Append(viewMenu, "&View")
        # viewMenu.Append(AppIDs.ID_DISP_ANIMATS, "&Animats", "Toggle display of animats", kind=wx.ITEM_CHECK)
        # viewMenu.Append(AppIDs.ID_DISP_OBJECTS, "&WorldObjects", "Toggle display of worldobjects", kind=wx.ITEM_CHECK)
        # viewMenu.Append(AppIDs.ID_DISP_TRAILS, "&Trails", "Toggle display of animat trails", kind=wx.ITEM_CHECK)
        # viewMenu.Append(AppIDs.ID_DISP_COLLISIONS, "&Collisions", "Toggle display of collisions", kind=wx.ITEM_CHECK)
        # viewMenu.Append(AppIDs.ID_DISP_SENSORS, "&Sensors", "Toggle display of sensor ranges", kind=wx.ITEM_CHECK)
        # viewMenu.Append(AppIDs.ID_DISP_MONITOR, "&Monitor", "Toggle display of monitor output", kind=wx.ITEM_CHECK)

        # Check view menu items
        # self.menuBar.Check(AppIDs.ID_DISP_ANIMATS, True)
        # self.menuBar.Check(AppIDs.ID_DISP_OBJECTS, True)
        # self.menuBar.Check(AppIDs.ID_DISP_TRAILS, True)
        # self.menuBar.Check(AppIDs.ID_DISP_COLLISIONS, True)
        # self.menuBar.Check(AppIDs.ID_DISP_SENSORS, True)
        # self.menuBar.Check(AppIDs.ID_DISP_MONITOR, True)

        # # World menu
        # worldMenu = wx.Menu()
        # self.m_pMenuBar.Append(worldMenu, "&World")
        # worldMenu.Append(AppIDs.ID_WORLD_NEXT, "&Next animat", "Selects the next animat")
        # worldMenu.Append(AppIDs.ID_WORLD_PREV, "&Previous animat", "Selects the previous animat")
        # worldMenu.Append(AppIDs.ID_WORLD_2D, "&2D Simulation", "Changes World to 2D")
        # worldMenu.Append(AppIDs.ID_WORLD_3D, "&3D Simulation", "Changes World to 3D")

        self.SetMenuBar(self.menuBar)


    def CreateSimMenus(self):
        """
        Create simulation menus for loading, saving, and starting simulations.

        This method dynamically creates menus and submenus for loading, saving, and starting simulations.
        It populates these menus based on the simulation names and contents available in the BeastFrame.

        Returns:
        -------
        None
        """
        pMenu = wx.Menu()
        pSubMenuLoad = wx.Menu()
        pSubMenuSave = wx.Menu()

        if self.currentSimulation is not None:
            simulation = self.currentSimulation

            # Populate menu for starting simulations
            for n in range(min(10, len(self.simulationNames))):
                pMenu.Append(AppIDs.ID_FILE_STARTSIM0 + n, f"Start simulation {self.simulationNames[n]}: ")

            # Populate submenus for loading and saving simulations
            for n, (name, obj) in enumerate(simulation.GetContents().items()):
                if n < 10:
                    strName = name
                    pSubMenuLoad.Append(AppIDs.ID_FILE_LOAD0 + n, strName, "Load saved " + strName)
                    pSubMenuSave.Append(AppIDs.ID_FILE_SAVE0 + n, strName, "Save current " + strName)

            pMenu.Append(AppIDs.ID_FILE_LOAD, "&Load", pSubMenuLoad, "Load saved...")
            pMenu.Append(AppIDs.ID_FILE_SAVE, "&Save", pSubMenuSave, "Save current...")
        else:
            pMenu.Append(AppIDs.ID_FILE_LOAD, "&Load", "Load saved simulation data")

        pMenu.Append(wx.ID_EXIT, "E&xit")

        # Remove existing 'File' menu and insert the newly created 'File' menu
        if self.menuBar.GetMenuCount() > 0:
            self.menuBar.Replace(0, pMenu, "&File")
        else:
            self.menuBar.Append(pMenu, "&File")

    def CreateLogWindow(self):
        """
        Create a log window for displaying logs or messages.

        This method creates a wxFrame object as the log window, with a wxTextCtrl
        widget for displaying log messages. The log window is positioned just below
        the BeastFrame, and its size is based on the width of the BeastFrame and a
        fixed height.
        """
        # Create a wxFrame as the log window
        myPos, mySize = self.GetPosition(), self.GetSize()

        self.logWindow = BeastLogWindow(self)
        self.logWindow.SetPosition(wx.Point(myPos.x + mySize.GetWidth() + 10, myPos.y))
        self.logWindow.SetSize(400, mySize.height)
        self.logWindow.Show(True)

    def CreateWorldCanvas(self, world: World):
        """
        Create a world canvas for displaying the simulation world.

        This method initializes a WorldGLCanvas, which is responsible for rendering
        the simulation world. It sets the canvas size based on the client size of
        the BeastFrame and triggers the OnSize event handler to properly adjust
        the rendering.
        """
        # Create a WorldGLCanvas with the simulation world and the BeastFrame as parent
        self.worldCanvas = WorldGLCanvas(self, self.GetClientSize(), world)
        event = wx.SizeEvent(self.GetClientSize())
        self.worldCanvas.OnSize(event)

    def BindEvents(self):
        """
        Run events
        """
        self.Bind(wx.EVT_CLOSE, self.OnExit)

        for item in self.demoMenu.GetMenuItems():
            self.Bind(wx.EVT_MENU, self.OnStartDemo, item)

        self.Bind(wx.EVT_MENU, self.OnLoad, id=AppIDs.ID_FILE_LOAD)
        self.Bind(wx.EVT_MENU, self.OnSave, id=AppIDs.ID_FILE_SAVE)

        if self.mySimulation:
            self.Bind(wx.EVT_MENU, self.OnStartMySimulation, id=AppIDs.ID_FILE_STARTMYSIM)

        self.Bind(wx.EVT_MENU, self.OnTglPaused, id=AppIDs.ID_SIM_PAUSE)
        self.Bind(wx.EVT_MENU, self.OnTglResume, id=AppIDs.ID_SIM_RESUME)
        self.Bind(wx.EVT_MENU, self.OnFast, id=AppIDs.ID_SIM_FAST)
        self.Bind(wx.EVT_MENU, self.OnReset, id=AppIDs.ID_SIM_RESET)
        self.Bind(wx.EVT_MENU, self.OnAbout, id=AppIDs.ID_HELP_ABOUT)
        self.Bind(wx.EVT_SIZE, self.OnSize)

        # TODO: Implement rendering options (Don't see the point)
        # self.Bind(wx.EVT_MENU, self.OnTglAnimats, id=AppIDs.ID_DISP_ANIMATS)
        # self.Bind(wx.EVT_MENU, self.OnTglObjects, id=AppIDs.ID_DISP_OBJECTS)
        # self.Bind(wx.EVT_MENU, self.OnTglTrails, id=AppIDs.ID_DISP_TRAILS)
        # self.Bind(wx.EVT_MENU, self.OnTglCollisions, id=AppIDs.ID_DISP_COLLISIONS)
        # self.Bind(wx.EVT_MENU, self.OnTglSensors, id=AppIDs.ID_DISP_SENSORS)
        # self.Bind(wx.EVT_MENU, self.OnTglMonitor, id=AppIDs.ID_DISP_MONITOR)

        # TODO: Make animats selectable (what's the usecase?)
        # self.Bind(wx.EVT_MENU, self.OnNextAnimat, id=AppIDs.ID_WORLD_NEXT)
        # self.Bind(wx.EVT_MENU, self.OnPrevAnimat, id=AppIDs.ID_WORLD_PREV)
        # self.Bind(wx.EVT_MENU, self.OnWorld3D, id=AppIDs.ID_WORLD_3D)
        # self.Bind(wx.EVT_MENU, self.OnWorld2D, id=AppIDs.ID_WORLD_2D)

    #================================================================================================
    # Everything that happens when simulation is started
    #================================================================================================

    def StartSimulation(self, simuluation: Simulation):
        """
        Starts a new simulation.

        Parameters:
        - nSim (int): Index of the simulation to start.

        Returns:
        None
        """
        self.initialSimulationCopy = copy.deepcopy(simuluation)

        if self.currentThread is not None:
            self.KillSimulationThread()

        self.currentSimulation = simuluation

        # Initialize simulation if it has not been loaded
        if not self.currentSimulation.hasBeenLoaded:
            self.currentSimulation.Init()

        # Tell GLCanvas about the world
        self.CreateWorldCanvas(self.currentSimulation.theWorld)
        # Adds simulation output to logger window
        self.currentSimulation.logger.addHandler(self.logWindow.handler)
        # Clear log window
        self.logWindow.logCtrl.Clear()

        # Create pause event that is shared between main and rendering thread (defaults to False)
        self.pauseEvent = threading.Event()
        # Create event to allow simulations to run fast by turning the rendering off
        self.renderSim = threading.Event()
        # Create event that kills thread
        self.killThread = threading.Event()
        # Run simulation in separate thread to avoid GUI freeze
        self.currentThread = threading.Thread(target=self.RunSimulation,
                                              args=(self.pauseEvent, self.renderSim, self.killThread))
        self.currentThread.daemon = True
        self.currentThread.start()

    #================================================================================================
    # Simulation thread/rendering loop
    #================================================================================================

    def RunSimulation(self, pauseEvent: threading.Event, renderSim: threading.Event, killThread: threading.Event):
        """
        Runs simulation
        """
        # Take ownership of context can display empty world
        self.worldCanvas.SetCurrent(self.worldCanvas.context)
        time.sleep(0.5)

        if not self.currentSimulation.hasBeenLoaded:
            # If simulation has not been loaded, we begin
            self.currentSimulation.BeginSimulation()
        else:
            # Else, we resume simulation
            self.currentSimulation.ResumeSimulation()

        # Clear pause event
        pauseEvent.set()
        # Set rendering
        renderSim.set()

        # This is the rendering loop: Update returns False until all runs are completed. A run consists of
        # 'self.m_pSimulation.generations' numbers of generations and each generation consists
        # of 'self.m_pSimulation.assessments' assessments, and each assessment runs for 'self.m_pSimulation.timeSteps'
        while True:

            # Pauses rendering loop if pauseEvent is set
            pauseEvent.wait()
            # Kills thread
            if killThread.is_set():
                break

            startTime = time.time()

            # Update simulation
            complete = self.currentSimulation.Update()

            # If renderSim is False, we run simulation without rendering to speed things up
            if renderSim.is_set():
                # Render the updated world
                wx.CallAfter(self.worldCanvas.Display)
                # Sleep if rendering faster than desired frame rate
                sleepTime = max(0, 1.0 / self.fps - (time.time() - startTime))
                time.sleep(sleepTime)

            if complete:
                break

            self.worldCanvas.SetCurrent(self.worldCanvas.context)

        self.worldCanvas.CleanUp()

        return

    #================================================================================================
    # Clean up simulation thread
    #================================================================================================

    def KillSimulationThread(self):

        self.killThread.set()
        self.Unpause()
        self.currentThread.join()
        time.sleep(1)
        self.DestroyWorldCanvas()


    def DestroyWorldCanvas(self):

        self.worldCanvas.Destroy()
        self.worldCanvas = None
        self.currentSimulation = None
        self.currentThread = None

    #================================================================================================
    # Event handlers
    #================================================================================================

    def OnSize(self, event: wx.SizeEvent):
        """
        Event handler for resizing the frame.

        This method is called when the frame is resized. It refreshes the frame
        to ensure proper rendering.

        Parameters:
        -----------
        event : wx.SizeEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """

        # Resize GLCanvas if its exists
        if self.worldCanvas is not None:
            if not self.paused:
                self.Pause()

            self.worldCanvas.OnSize(event)

        self.Refresh()

    def OnStartMySimulation(self, event: wx.CommandEvent):

        self.StartSimulation(self.mySimulation)


    def OnStartDemo(self, event: wx.CommandEvent):
        """
        Event handler for starting a simulation.

        This method is called when a menu item for starting a simulation is clicked.
        It extracts the simulation index from the event ID and then calls the StartSimulation
        method with the appropriate argument.

        Parameters:
        -----------
        event : wx.CommandEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """
        guiName = self.menuBar.FindItemById(event.GetId()).GetItemLabel()
        simulation = self.simulationClass[self.simulationNames.index(guiName[1:])]()
        self.StartSimulation(simulation)

    def OnLoad(self, event: wx.CommandEvent):
        """
        Event handler for loading a simulation state.

        This method is called when a menu item for loading a simulation is clicked.
        It retrieves the simulation object and iterates through its contents to find
        the corresponding simulation index based on the event ID. It then prompts
        the user to select a population file to load. If a file is selected,
        it attempts to load the population data into the simulation.

        Parameters:
        -----------
        event : wx.CommandEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """
        self.Pause()

        openFileDialog = wx.FileDialog(self, "Open", str(DATA_DIR), "",
            "Binary files (*.pkl)|*.pkl", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if openFileDialog.ShowModal() == wx.ID_OK:
            filepath = openFileDialog.GetPath()

            try:
                with open(filepath, 'rb') as f:
                    simulation = pickle.load(f)
            except Exception as e:
                msg = f"Failed to load simulation state {Path(filepath).stem}:" + str(e)
                self.logWindow.logMessage(msg)

            self.StartSimulation(simulation)

        openFileDialog.Destroy()


    def OnSave(self, event: wx.CommandEvent):
        """
        Handles the event for saving the simulation.

        Parameters:
        - event (wx.CommandEvent): The event object.

        Returns:
        None
        """
        if self.currentSimulation is None:
            return

        self.Pause()

        saveFileDialogue = wx.FileDialog(self, message="Save simulation state", defaultDir=str(DATA_DIR),
            defaultFile="", wildcard="Pickle files (*.pkl)|*.pkl",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if saveFileDialogue.ShowModal() == wx.ID_OK:
            self.currentSimulation.Serialize(saveFileDialogue.GetPath() + '.pkl')

        saveFileDialogue.Destroy()

        self.Unpause()

    def OnExit(self, event: wx.CloseEvent):

        """
        Event handler for exiting the application.

        This method is called when the exit menu item or button is clicked.
        It stops the timer (if it exists) and destroys the frame, effectively
        closing the application.

        Parameters:
        -----------
        event : wx.CommandEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """
        if self.currentThread is not None:
            self.KillSimulationThread()

        if self.logWindow is not None:
            self.logWindow.Destroy()

        self.Destroy()

    def OnTglPaused(self, event):
        """
        Event handler for toggling the pause state of the simulation.

        This method is called when the pause menu item or button is clicked.
        It toggles the pause state of the simulation by calling the Pause()
        or Unpause() method accordingly.

        Parameters:
        -----------
        event : wx.CommandEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """
        if not self.paused:
            self.Pause()


    def Pause(self):
        """
        Pause the simulation.

        This method pauses the simulation by stopping the timer and pausing the stopwatch.
        It also updates the menu item label and help string accordingly.

        Returns:
        -------
        None
        """
        if self.currentSimulation is None:
            return

        if self.paused:
            return
        else:
            # Set pauseEvent
            self.pauseEvent.clear()
            self.paused = True


    def OnTglResume(self, event):
        """
        Event handler for toggling the resume state of the simulation.

        This method is called when the resume menu item or button is clicked.
        It toggles ir resumes the simulation by calling the Unpause() method
        assuming that the simulation is paused.

        Parameters:
        -----------
        event : wx.CommandEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """
        if self.paused:
            self.Unpause()

    def OnReset(self, event):
        """
        Event handler for resetting the simulation.

        This method is called when the reset menu item or button is clicked.
        It starts the simulation with the current simulation ID.

        Parameters:
        -----------
        event : wx.CommandEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """
        if self.initialSimulationCopy is not None:
            self.StartSimulation(self.initialSimulationCopy)

    def Unpause(self):
        """
        Unpause the simulation.

        This method resumes the simulation by starting the timer.
        It also updates the menu item label and help string accordingly.

        Returns:
        -------
        None
        """
        if self.currentSimulation is None:
            return

        if not self.paused:
            return
        else:
            self.pauseEvent.set()
            self.paused = False

    def OnAbout(self, event):
        """
        Event handler for displaying information about the application.

        This method is called when the about menu item or button is clicked.
        It displays information about the application in a message box.

        Parameters:
        -----------
        event : wx.CommandEvent
            The event object containing information about the event.

        Returns:
        -------
        None
        """
        wx.MessageBox("BEAST",
                      "Bioinspired Evolutionary Agent Simulation Toolkit\nVersion 0.00001",
                      wx.ICON_INFORMATION)

    def OnFast(self, event: wx.CommandEvent):
        """
        Turn off rendering to run simulations high speed
        """
        if self.renderSim.is_set():
            self.renderSim.clear()
        else:
            self.renderSim.set()

    def HighSpeed(self):
        """
        Run the simulation at high speed.

        This method runs the simulation at high speed, updating the simulation
        without waiting for real-time constraints. It also displays a progress dialog
        showing the progress of the simulation.

        Returns:
        -------
        None
        """
        if self.currentSimulation is None:
            return

        self.Render.clear()


#================================================================================================
# Not implemented events
#================================================================================================
    #
    # def OnTglAnimats(self, event):
    #     """
    #     Event handler for toggling the display of animats.
    #
    #     This method is called when the corresponding menu item or button for
    #     toggling the display of animats is clicked. It toggles the display
    #     of animats on the world canvas and updates the menu item's check state
    #     accordingly.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     self.worldCanvas.Toggle(self.currentSimulation.GetWorld().WORLD_DISPLAY_TYPE.DISPLAY_ANIMATS)
    #     self.menuBar.Check(AppIDs.ID_DISP_ANIMATS, event.IsChecked())
    #
    # def OnTglObjects(self, event):
    #     """
    #     Event handler for toggling the display of world objects.
    #
    #     This method is called when the corresponding menu item or button for
    #     toggling the display of world objects is clicked. It toggles the display
    #     of world objects on the world canvas and updates the menu item's check state
    #     accordingly.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     self.worldCanvas.Toggle(self.currentSimulation.GetWorld().WORLD_DISPLAY_TYPE.DISPLAY_WORLDOBJECTS)
    #     self.menuBar.Check(AppIDs.ID_DISP_OBJECTS, event.IsChecked())
    #
    # def OnTglTrails(self, event):
    #     """
    #     Event handler for toggling the display of animat trails.
    #
    #     This method is called when the corresponding menu item or button for
    #     toggling the display of animat trails is clicked. It toggles the display
    #     of animat trails on the world canvas and updates the menu item's check state
    #     accordingly.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     self.worldCanvas.Toggle(self.currentSimulation.GetWorld().WORLD_DISPLAY_TYPE.DISPLAY_TRAILS)
    #     self.menuBar.Check(AppIDs.ID_DISP_TRAILS, event.IsChecked())
    #
    # def OnTglCollisions(self, event):
    #     """
    #     Event handler for toggling the display of collisions.
    #
    #     This method is called when the corresponding menu item or button for
    #     toggling the display of collisions is clicked. It toggles the display
    #     of collisions on the world canvas and updates the menu item's check state
    #     accordingly.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     self.worldCanvas.Toggle(self.currentSimulation.GetWorld().WORLD_DISPLAY_TYPE.DISPLAY_COLLISIONS)
    #     self.menuBar.Check(AppIDs.ID_DISP_COLLISIONS, event.IsChecked())
    #
    # def OnTglSensors(self, event):
    #     """
    #     Event handler for toggling the display of sensor ranges.
    #
    #     This method is called when the corresponding menu item or button for
    #     toggling the display of sensor ranges is clicked. It toggles the display
    #     of sensor ranges on the world canvas and updates the menu item's check state
    #     accordingly.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     self.worldCanvas.Toggle(self.currentSimulation.GetWorld().WORLD_DISPLAY_TYPE.DISPLAY_SENSORS)
    #     self.menuBar.Check(AppIDs.ID_DISP_SENSORS, event.IsChecked())
    #
    # def OnTglMonitor(self, event):
    #     """
    #     Event handler for toggling the display of monitor output.
    #
    #     This method is called when the corresponding menu item or button for
    #     toggling the display of monitor output is clicked. It toggles the display
    #     of monitor output on the world canvas and updates the menu item's check state
    #     accordingly.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     self.worldCanvas.Toggle(self.currentSimulation.GetWorld().WORLD_DISPLAY_TYPE.DISPLAY_MONITOR)
    #     self.menuBar.Check(AppIDs.ID_DISP_MONITOR, event.IsChecked())
    #
    #
    # def OnNextAnimat(self, event):
    #     """
    #     Event handler for selecting the next animat.
    #
    #     This method is called when the next animat menu item or button is clicked.
    #     It selects the next animat in the simulation's world.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     if self.currentSimulation is None:
    #         return
    #     self.currentSimulation.GetWorld().OnSelectNext()
    #
    # def OnPrevAnimat(self, event):
    #     """
    #     Event handler for selecting the previous animat.
    #
    #     This method is called when the previous animat menu item or button is clicked.
    #     It selects the previous animat in the simulation's world.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     if self.currentSimulation is None:
    #         return
    #     self.currentSimulation.GetWorld().OnSelectPrevious()
    #
    # # Functions created for 2D and 3D worlds
    # def OnWorld3D(self, event):
    #     """
    #     Event handler for switching to a 3D world.
    #
    #     This method is called when the 3D world menu item or button is clicked.
    #     It switches the simulation's world to a 3D representation.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     if self.currentSimulation is None:
    #         return
    #     self.currentSimulation.GetWorld().World3D()
    #
    # def OnWorld2D(self, event):
    #     """
    #     Event handler for switching to a 2D world.
    #
    #     This method is called when the 2D world menu item or button is clicked.
    #     It switches the simulation's world to a 2D representation.
    #
    #     Parameters:
    #     -----------
    #     event : wx.CommandEvent
    #         The event object containing information about the event.
    #
    #     Returns:
    #     -------
    #     None
    #     """
    #     if self.currentSimulation is None:
    #         return
    #     self.currentSimulation.GetWorld().World2D()

