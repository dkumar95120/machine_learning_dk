import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
tolerance = .05
class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, optimized=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.optimized = optimized # if the agent is optimized

        self.Q = {}

        # Set any additional class parameters as needed
        # Initialize basic parameters of the Q-learning equation
        # Trials for plotting
        self.waypoint = None
        self.moves = 0
        self.steps = 0
        self.success = 0
        self.failure = 0
        self.trials = 0


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        self.trials += 1
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
        # Update epsilon using a decay function
            if self.optimized:
                self.epsilon = 1.0/math.sqrt(1.0+self.trials)
            else:  
               self.epsilon -= .05   # linear decay


        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        self.waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        # Set 'state' as a tuple of relevant data for the agent  
        state = (self.waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        return state

    def printQ(self):
        print "Q Matrix"
        print "{0:40},".format(' waypoint,    light, oncoming, l-traffic'),
        for action in self.valid_actions:
            if action:
                print "{0:>10},".format(action),
            else:
                print "{0:>9},".format(action),
        print 'Best Action\n'
        for state in self.Q:
            _, best_action = self.get_maxQ(state)
            waypoint, light, oncoming, left_traffic = state
            print "{0:>9},{1:>9},{2:>9},{3:>9},".format(waypoint, light, oncoming, left_traffic),
            for action in self.valid_actions:
                print "{0:10.3f},".format(self.Q[state][action]),
            print "{0:>10}".format(best_action)

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        # Calculate the maximum Q-value of all actions for a given state
        maxQ = -1.e10
        preferred_action = None
        for action, qval in self.Q[state].iteritems():
            if qval > maxQ :
                maxQ = qval
                preferred_action = action

        return maxQ, preferred_action 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning and state not in self.Q:
            print "new state:", state
            self.Q[state] ={}
            for action in self.valid_actions:
                self.Q[state][action] = 0.0

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        action = None

        # When not learning, choose a random action, or
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if self.learning:
            p = random.random()
            if p < self.epsilon:
                action = random.choice(self.valid_actions)
            else:
                _, action = self.get_maxQ(state)
        else:
            action = random.choice(self.valid_actions)
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        print "state = {0:35} action = {1:10} reward = {2:6.3f}".format(state, action, reward)

        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        q = self.Q[state][action]
        self.Q[state][action] = (1 - self.alpha)*q + self.alpha*reward
        #print "Q(current={0:6.3f}, previous={1:6.3f})".format(self.Q[state][action], q)
        
        # update stats
        self.moves+=1
        self.steps+=1

        if reward >5:
            self.success += 1
        elif reward<0:
            self.failure +=1
        
        failure_rate = float(self.failure)/float(self.moves)

        #print "success={0:5d}, steps={1:5d}, failure_rate={2:5.3f}".format(self.success, self.steps, failure_rate)
        
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action, self.waypoint) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    optimized = True
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning=True, optimized=optimized, epsilon=1.0, alpha=.6)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=.01, log_metrics=True, display=False, optimized=optimized)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(tolerance=tolerance, n_test=100)
    agent.printQ()


if __name__ == '__main__':
    run()
