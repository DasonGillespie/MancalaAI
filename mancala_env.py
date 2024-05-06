import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)

    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    env = wrappers.AssertOutOfBoundsWrapper(env)
    #env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "mancala_v1"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {"player_0" : 0, "player_1" : 1}
        self._action_space = Discrete(6)
        self._observation_space = Discrete(14)
        self.render_mode = render_mode
        #a dictionary containing the holes directly opposite on the board
        self.opposites = {0: 12, 1: 11,
                          2: 10, 3: 9,
                          4: 8, 5: 7,
                          7: 5, 8: 4,
                          9: 3, 10: 2,
                          11: 1, 12: 0}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(14)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(6, start=0, seed=42)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return

        if len(self.agents) == 2:
            s1 = "\n    PLAYER_1  SIDE   \n"
            n1 = "  (5)(4)(3)(2)(1)(0)  \n"
            l1 = "+--------------------+\n"
            l2 = "|  " + str(self.state[12]) + "  " + str(self.state[11]) + "  " + str(self.state[10]) + "  " + str(self.state[9]) + "  " + str(self.state[8]) + "  " + str(self.state[7]) + "  |\n"
            if (self.state[13] > 9 and self.state[6] <= 9) or (self.state[13] <= 9 and self.state[6] > 9):
                l3 = "|" + str(self.state[13]) + "                 " + str(self.state[6]) + "|\n"
            elif self.state[13] > 9 and self.state[6] > 9:
                l3 = "|" + str(self.state[13]) + "                " + str(self.state[6]) + "|\n"
            else:
                l3 = "|" + str(self.state[13]) + "                  " + str(self.state[6]) + "|\n"
            
            l4 = "|  " + str(self.state[0]) + "  " + str(self.state[1]) + "  " + str(self.state[2]) + "  " + str(self.state[3]) + "  " + str(self.state[4]) + "  " + str(self.state[5]) + "  |\n"
            n2 = "  (0)(1)(2)(3)(4)(5)  \n"
            s2 = "    PLAYER_0  SIDE   \n"
            string = s1 + n1 + l1 + l2 + l3 + l4 + l1 + n2 + s2
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        # observation of one agent is the previous state of the other
        return self.state

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        #rewards will not be used
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        #to be used by agent
        self.extra_turn = False
        self.game_over = False
        self.goal = {"player_0": 6, "player_1": 13}
        self.opp_goal = {"player_0": 13, "player_1": 6}
        self.render_bool = True
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        #data analysis
        self.data_testing = False
        self.final_score = {agent: 0 for agent in self.agents}
        self.hard_off = False

        if options == True:
            self.hard_off = True

        if not self.hard_off:
            self.render()

    def step(self, action):

        #don't touch this
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        if action < 0 or action > 5:
            print("invalid action")
            return
        
        #--begin--
        agent = self.agent_selection

        #turn off extra turn
        self.extra_turn = False

        #apply action according to player and respective corresponding board numbers
        if agent == "player_0":
            count = self.state[action]
            current = action

            #remove stones from hole
            self.state[current] = 0

            #place stones in consecutive holes
            current += 1
            current = current % 14 #make sure to wrap around if necessary
            for i in range(count):
                self.state[current] += 1
                current += 1
                if current == 13: #skip the opponent's goal
                    current += 1
                current = current % 14 #wrap

            last_hole = current -1 #current was advanced past the last hole placed

            #extra turn
            if last_hole == 6:
                self.extra_turn = True

            #capture opponents stones if last_hole...
            #was empty and on current player's side
            if self.state[last_hole] == 1 and last_hole >= 0 and last_hole <= 5:
                opp = self.opposites[last_hole]
                opp_stones = self.state[opp]
                if opp_stones != 0:
                    self.state[6] += opp_stones + 1
                    self.state[opp] = 0
                    self.state[last_hole] = 0

            #check if the game is over
            total = 0
            for i in range(6):
                total += self.state[i]
            if total == 0: #game is over, this side is empty, opponent claims remaining stones
                claim = 0
                for i in range(6):
                    claim += self.state[i + 7]
                    self.state[i + 7] = 0
                self.state[13] += claim
                self.game_over = True
                self.terminations = {agent: True for agent in self.agents}

            #game might be over as a result of a capture
            #check the opponent's side in that case
            total = 0
            for i in range(6):
                total += self.state[i + 7]
            if total == 0: #game is over, this side is empty, opponent claims remaining stones
                claim = 0
                for i in range(6):
                    claim += self.state[i]
                    self.state[i] = 0
                self.state[6] += claim
                self.game_over = True
                self.terminations = {agent: True for agent in self.agents}
                

        elif agent == "player_1":
            count = self.state[action + 7]
            current = action + 7

            self.state[current] = 0

            current += 1
            current = current % 14 
            for i in range(count):
                self.state[current] += 1
                current += 1
                if current == 6: 
                    current += 1
                current = current % 14 

            last_hole = current -1 
            if last_hole == -1:
                last_hole = 13
            
            if last_hole == 13:
                self.extra_turn = True

            if self.state[last_hole] == 1 and last_hole >= 7 and last_hole <= 12:
                opp = self.opposites[last_hole]
                opp_stones = self.state[opp]
                if opp_stones != 0:
                    self.state[13] += opp_stones + 1
                    self.state[opp] = 0
                    self.state[last_hole] = 0

            #check if the game is over
            total = 0
            for i in range(6):
                total += self.state[i + 7]
            if total == 0: #game is over, this side is empty, opponent claims remaining stones
                claim = 0
                for i in range(6):
                    claim += self.state[i]
                    self.state[i] = 0
                self.state[6] += claim
                self.game_over = True
                self.terminations = {agent: True for agent in self.agents}

            #check opponent's side (possible capture)
            total = 0
            for i in range(6):
                total += self.state[i]
            if total == 0: #game is over, this side is empty, opponent claims remaining stones
                claim = 0
                for i in range(6):
                    claim += self.state[i + 7]
                    self.state[i + 7] = 0
                self.state[13] += claim
                self.game_over = True
                self.terminations = {agent: True for agent in self.agents}

        else:
            print("Error: agent does not equal player_0 or player_1 in step()")
            return
   
        #advance agent to the next agent
        if not self.extra_turn:
            self.agent_selection = self._agent_selector.next()

        #dont touch this
        self._accumulate_rewards()

        if self.render_bool and not self.hard_off:
            self.render()

            if self.game_over:
                if self.state[6] > self.state[13]:
                    print("---!!!Player_0 WINS!!!---")
                elif self.state[13] > self.state[6]:
                    print("---!!!Player_1 WINS!!!---")
                else:
                    print("---!!!GAME TIED!!!---")

                print("\n FINAL SCORE")
                print("-------------")
                print("P0:", self.state[6], "P1:", self.state[13], "\n")

        if self.extra_turn == True and self.render_bool and not self.hard_off:
            print(self.agent_selection, "gained an extra turn\n")

        #data analysis
        self.final_score["player_0"] = self.state[6]
        self.final_score["player_1"] = self.state[13]

        self.render_bool = True

    #Agent Functions
    def extra_turn(self):
        return self.extra_turn
    
    def ACTIONS(self):
        actions = []
        agent = self.agent_selection
        if agent == "player_0":
            for i in range(6):
                if self.state[i] != 0:
                    actions.append(i)
        elif agent == "player_1":
            for i in range(6):
                if self.state[i + 7] != 0:
                    actions.append(i)
        else:
            print("Error: agent does not equal player_0 or player_1 in ACTIONS()")
            return []
        
        return actions
    
    def GAME_OVER(self):
        return self.game_over
    
    def current_agent(self):
        return self.agent_selection

    def set_render(self, render):
        self.render_bool = render

    def set_hard_off(self, b):
        self.hard_off = b

    def get_final_score(self):
        return self.final_score
