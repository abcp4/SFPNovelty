import environment as env
import renderer
import robot
import utils
import wall

from fitness_function_factory import FitnessFunctionFactory

class MazeSimulator():

    def __init__(self, render, xml_file):

        if render:
            self.renderer = renderer.Renderer(600, 500, 'Maze Simulator')

        self.env = env.Environment(xml_file)
        self.pois_reached = [False] * len(self.env.pois)
        self.reached_goal = False

    def reset(self):
        self.env.reset()
        self.pois_reached = [False] * len(self.env.pois)
        self.reached_goal = False
    
    def get_possible_actions(self):
        return [[0, 0.4, 0],[0.05, 0.2, 0],[0, 0.2, 0.05]]

    def step_robot(self, outputs, time_step):
        """
        Step the robot in time where "outputs" is the action vector.
        """
        self.env.robot.decide_action(outputs, time_step)
        self.env.robot.update_position()
        # Check for collision
        did_collide = self.check_collisions()
        if did_collide:
            self.env.robot.undo()
        # Update RangeFinders
        self.env.robot.update_rangefinders(self.env.walls)
        self.env.robot.update_radars(self.env.goal)

    def step(self, action, time_delta,repeat = 20):
        for i in range(repeat):
            finder_obs = self.env.robot.get_rangefinder_observations()
            radar_obs = self.env.robot.get_radar_observations()
            done = self.reached_goal

            self.step_robot(action, time_delta)
            done = self.update_pois()

        return finder_obs, radar_obs, done


    def check_collisions(self):
        for wall in self.env.walls:
            if utils.collide(wall, self.env.robot):
                return True
        
    def update_pois(self):
        """
        Update the reached pois and if the robot has reached the goal.
        Return whether or not the robot has reached the goal.
        """
        if utils.robot_distance_from_goal(self.env) < 35.0:
            self.reached_goal = True
            return True
        else:
            for i in range(0, len(self.env.pois)):
                if utils.robot_distance_from_poi(self.env, i) < 20.0:
                    self.pois_reached[i] = True
            return False

    def render(self):
        self.renderer.render(self.env)

    #def evaluate_fitness(self,debug = False):
    #    factory = FitnessFunctionFactory()
    #    fitness_function = factory.get_fitness_function('hardmaze')
    #    return fitness_function(self.pois_reached, self.env, self.reached_goal)
    
    def evaluate_fitness(self):
        return -utils.robot_distance_from_goal(self.env) 




    def quit(self):
        pygame.quit()