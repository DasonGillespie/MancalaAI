import mancala_env as mancala
import copy
import random
import math

#------
#GLOBAL
#------

max_depth = 5

#----------
#END GLOBAL
#----------

def agent_function(env, agent):
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = None
        val, action = MINIMAX(env, 0)
        if action is None:
            print("MINIMAX did not find an action. A random action is being taken.")
            action = random.choice(env.legal_moves())
    return action

# MINIMAX

def MINIMAX(s, depth):

    #find out who the current player is
    #necessary because the last action might have earned an extra turn
    agent = s.current_agent()
    player = False
    if agent == "player_0":
        player = True

    if depth == max_depth or s.GAME_OVER(): #max_depth is a global variable
        return EVALUATE(s, player), None
    
    if player: #(MAX)
        best_val = -math.inf
        best_action = None
        actions = s.ACTIONS()
        for a in reversed(actions):
            s2 = RESULT(s, a)
            new_val, _ = MINIMAX(s2, depth + 1)
            if new_val > best_val:
                best_val = new_val
                best_action = a
        return best_val, best_action
    
    else: #not player (MIN)
        best_val = math.inf
        best_action = None
        actions = s.ACTIONS()
        for a in reversed(actions):
            s2 = RESULT(s, a)
            new_val, _ = MINIMAX(s2, depth + 1)
            if new_val < best_val:
                best_val = new_val
                best_action = a
        return best_val, best_action

#!!!IMPORTANT!!! EVALUATE happens before the action step
#if the game is over, it is because the previous player won
def EVALUATE(s, player):
    total = 0
    if s.GAME_OVER():
        if player:
            total += 5
        else:
            total -= 5
    return total

def RESULT(s, a):
    env1 = copy.deepcopy(s)
    env1.set_render(False)
    env1.step(a)
    return env1

def median_sort(lst):
    median = sorted(lst)[len(lst) // 2]
    return sorted(lst, key=lambda x: abs(x - median))