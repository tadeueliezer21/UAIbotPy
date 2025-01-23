from utils import *
import numpy as np
from graphics.meshmaterial import *
from uaibot.simobjects.cylinder import *

class Pedestrian(Cylinder):
    """
    Initialize a pedestrian agent.

    Parameters
    ----------
    ra : numpy array
        Position of the agent.
    g : numpy array
        Position of the agents goal.
    name : str, optional
        Name of the agent. The default is "pedestrian".
    color : str, optional
        Color of the agent. The default is "CornflowerBlue".
    height : float, optional
        Height of the agent. The default is 1.7.
    radius : float, optional
        Radius of the agent. The default is 0.3.
    va : numpy array, optional
        Velocity of the agent. The default is np.array([[0],[0]]).
    va0 : float, optional
        Desired velocity magnitude. The default is 1.3.
    ta : float, optional
        Relaxation time. The default is 1.
    ma : float, optional
        Agent mass. The default is 1.
    a_beta : float, optional
        Repulsive force amplitude relative to other pedestrians. The default is 5.
    b_beta : float, optional
        Repulsive force decay relative to other pedestrians. The default is 0.1.
    a_i : float, optional
        Repulsive force amplitude relative to obstacles. The default is 8.
    b_i : float, optional
        Repulsive force decay relative to obstacles. The default is 0.2.
    lambda_a : float, optional
        Directional sensitivity. The default is 1.
    """

    #######################################
    # Attributes
    #######################################

    @property
    def ra(self):
        return self._ra

    @ra.setter
    def ra(self, value):
        self._ra = value

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, value):
        self._g = value

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, value):
        self._color = value
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        self._height = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def va(self):
        return self._va

    @va.setter
    def va(self, value):
        self._va = value

    @property
    def va0(self):
        return self._va0

    @va0.setter
    def va0(self, value):
        self._va0 = value

    @property
    def ta(self):
        return self._ta

    @ta.setter
    def ta(self, value):
        self._ta = value

    @property
    def ma(self):
        return self._ma

    @ma.setter
    def ma(self, value):
        self._ma = value

    @property
    def a_beta(self):
        return self._a_beta

    @a_beta.setter
    def a_beta(self, value):
        self._a_beta = value

    @property
    def b_beta(self):
        return self._b_beta

    @b_beta.setter
    def b_beta(self, value):
        self._b_beta = value

    @property
    def a_i(self):
        return self._a_i

    @a_i.setter
    def a_i(self, value):
        self._a_i = value

    @property
    def b_i(self):
        return self._b_i

    @b_i.setter
    def b_i(self, value):
        self._b_i = value

    @property
    def lambda_a(self):
        return self._lambda_a

    @lambda_a.setter
    def lambda_a(self, value):
        self._lambda_a = value

    @property
    def is_pedestrian(self):
        return True
    
    #######################################
    # Constructor
    #######################################

    def __init__(
        self,
        ra: np.array,
        g: np.array,
        name: str = "pedestrian",
        color: str = "CornflowerBlue",
        height: float = 1.7,
        radius: float = 0.3,
        va: np.array = np.array([[0], [0]]),
        va0: float = 1.3,
        ta: float = 1.0,
        ma: float = 1.0,
        a_beta: float = 8.0,
        b_beta: float = 0.2,
        a_i: float = 5.0,
        b_i: float = 0.3,
        lambda_a: float = 0.3,
    ):
        htm = np.eye(4)
        htm[0][3] = ra[0]
        htm[1][3] = ra[1]
        htm[2][3] = height / 2

        super().__init__(htm=htm, name=name, radius=radius, height=height, mass=1, color=color, opacity=1, mesh_material=None)
        self._ra = ra
        self._g = g
        self._name = name
        self._color = color
        self._height = height
        self._radius = radius
        self._va = va
        self._va0 = va0
        self._ta = ta
        self._ma = ma
        self._a_beta = a_beta
        self._b_beta = b_beta
        self._a_i = a_i
        self._b_i = b_i
        self._lambda_a = lambda_a

    def fag(self):
            """
            Compute the attractive force towards a goal.

            This function calculates the force required to steer an agent or object
            from its current velocity (va) toward a goal direction (ea0) at a
            desired velocity magnitude (va0). It uses a simple linear force model.
            """
            ea0 = (self._g - self._ra) / np.linalg.norm(self._ra - self._g)  # desired direction
            return self._ma * (1 / self._ta) * (self._va0 * ea0 - self._va)
    
    def fdaq(self, obstacle):
        """
        Compute the distance-dependent repulsive force.

        This function implements the exponential decay of repulsive force based on the
        distance vector d. The force magnitude decays exponentially with the norm of d,
        and its direction is along the unit vector of d.
        """
        d = obstacle.d(self._ra, self._radius)
        norm_d = np.linalg.norm(d)

        if obstacle.is_pedestrian:
            fdaq = self._a_beta * np.exp(-norm_d / self._b_beta) * (d / norm_d)
        else:
            fdaq = self._a_i * np.exp(-norm_d / self._b_i) * (d / norm_d)

        return fdaq
    
    def w(self, obstacle):
        """
        Compute the anisotropic weighting function for repulsive forces.

        This function introduces directional sensitivity to interactions
        based on the angle phi_aq, which is the angle between the agent's
        velocity vector and the direction of another entity. The parameter
        lambda_a modulates how much the agent is influenced by interactions
        from behind.
        """
        if self._va[0] == 0 and self._va[1] == 0:
            return 1 # if the agent is stationary, it is not influenced by interactions
        
        d = obstacle.d(self._ra, self._radius)
        phi_aq = np.arccos(
            np.dot(
                self._va.flatten() / np.linalg.norm(self._va),
                -d.flatten() / np.linalg.norm(d),
            )
        )

        return self._lambda_a + (1 - self._lambda_a) * (1 + np.cos(phi_aq)) / 2
    
    def d(self, rb, b_radius):
        """
        Compute the distance vector between two agents.
        This function calculates the distance vector between two agents
        based on their positions and radii.

        Parameters
        ----------
        rb : numpy array
            Position of the other agent.
        b_radius : float
            Radius of the other agent
        """

        d = (rb - self._ra) - (self._radius + b_radius) * (rb - self._ra) / np.linalg.norm(
            rb - self._ra
        )  # distance from obstacle to pedestrian
        return d
    

class ObstacleColumn(Cylinder):
    """
    Initialize a pedestrian agent.

    Parameters
    ----------
    ra : numpy array
        Position of the agent.
    name : str, optional
        Name of the agent. The default is "ObstacleColumn".
    color : str, optional
        Color of the agent. The default is "DarkGrey".
    height : float, optional
        Height of the agent. The default is 1.7.
    radius : float, optional
        Radius of the agent. The default is 0.3.
    va : numpy array, optional
        Velocity of the agent. The default is np.array([[0],[0]]).
    """

    #######################################
    # Attributes
    #######################################

    @property
    def ra(self):
        return self._ra

    @ra.setter
    def ra(self, value):
        self._ra = value

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, value):
        self._color = value
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        self._height = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def va(self):
        return self._va

    @va.setter
    def va(self, value):
        self._va = value

    @property
    def is_pedestrian(self):
        return False
    
    #######################################
    # Constructor
    #######################################

    def __init__(
        self,
        ra: np.array,
        name: str = "ObstacleColumn",
        color: str = "DarkGrey",
        height: float = 2.4,
        radius: float = 0.3,
        va: np.array = np.array([[0], [0]])
    ):
        htm = np.eye(4)
        htm[0][3] = ra[0]
        htm[1][3] = ra[1]
        htm[2][3] = height / 2

        super().__init__(htm=htm, name=name, radius=radius, height=height, mass=1, color=color, opacity=1, mesh_material=None)
        self._ra = ra
        self._name = name
        self._color = color
        self._height = height
        self._radius = radius
        self._va = va

    
    def d(self, rb, b_radius):
        """
        Compute the distance vector between two agents.
        This function calculates the distance vector between two agents
        based on their positions and radii.

        Parameters
        ----------
        rb : numpy array
            Position of the other agent.
        b_radius : float
            Radius of the other agent
        """

        d = (rb - self._ra) - (self._radius + b_radius) * (rb - self._ra) / np.linalg.norm(
            rb - self._ra
        )  # distance from obstacle to pedestrian
        return d