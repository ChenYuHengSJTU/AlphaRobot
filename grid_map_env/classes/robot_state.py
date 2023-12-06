class RobotState:
    """
    Class representing an robot's state.

    The robot occupies a square grid of size 2*2, where:
    - The 'row' and 'col' parameters represent the robot's coordinates.
    - The 'direction' parameter represents the robot's rot: 0 for west, 1 for north, 2 for east, and 3 for south.
    - The 'speed' parameter represents the robot's speed.
    """

    def __init__(self, row=0, col=0, direction=0, speed=0):
        """
        Constructor for the robot class.
        Member variables:
            row (int): an integer representing the row coordinate of the robot.
            col (int): an integer representing the column coordinate of the robot.
            direction (int): an integer representing the robot's direction, see Table 3 for its encoding.
            speed (int): an integer indicating the robot's speed, chosen from {0,1,2,3}.

        """

        self.row = row
        self.col = col
        self.direction = direction
        self.speed = speed

    def copy(self):
        """
        Deep copy function for RobotState.
        """
        return RobotState(row=self.row, 
                          col=self.col, 
                          direction=self.direction, 
                          speed=self.speed)