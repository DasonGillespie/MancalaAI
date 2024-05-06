import mancala_env as mancala
import random

def agent_function(env, agent):
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation or env.GAME_OVER():
        action = None
    else:
        action = None
        #val, action = MINIMAX(searchable_env, 0, True)
        #print("value of action being taken:", val)
        if action is None:
            available = env.ACTIONS()
            if len(available) > 0:
                action = random.choice(available)
    return action

