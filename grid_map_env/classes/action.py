class Action:
    """
    This is a class representing actions. Each action consists of two parameters: acceleration and rotation direction.

    - Acceleration can only be one of -1, 0, 1, representing deceleration by 1, maintaining constant speed, and acceleration by 1, respectively.
    - Rotation direction can only be one of -1, 0, 1, representing left turn, maintaining direction, and right turn, respectively.
    """

    def __init__(self, acc, rot):
        """
        Constructor for the Action class.
        Member variables: 
            acc (int): an integer representing acceleration. See Table 4 for its encoding.
            rot (int): an integer representing rotation direction. See Table 5 for its encoding.
        """
        self.acc = acc
        self.rot = rot

    def is_legal(self, robot_state):
        """
        This member function checks the validity of an action.
        - acc can only be -1, 0, or 1.
        - rot can only be -1, 0, or 1.
        - rotation is allowed only when the robot's speed is 0, and no acceleration or deceleration is allowed when rotation is performed.
        """
        if self.acc not in [-1, 0, 1]:
            return False
        if self.rot not in [-1, 0, 1]:
            return False

        if robot_state.speed != 0:
            if self.rot != 0:
                return False
        else:
            if self.acc != 0:
                if self.rot != 0:
                    return False
                
        return True
