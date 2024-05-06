import mancala_env as mancala
'''
env = mancala.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    print("---" + agent + "'s turn---\n")

    if termination or truncation:
        action = None
    else:
        #action = env.action_space(agent).sample()
        action = None
        available = env.ACTIONS()
        print("available actions:", available)
        while action not in available:
            action = int(input("action? "))
            if action not in available:
                print("Bad choice. Try again.")

    env.step(action)
env.close()
'''

def agent_function(env, agent):
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation or env.GAME_OVER():
        action = None
    else:
        action = None
        available = env.ACTIONS()
        print("available actions:", available)
        while action not in available:
            action = int(input("action? "))
            if action not in available:
                print("Bad choice. Try again.")
        
    return action