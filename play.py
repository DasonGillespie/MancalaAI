import mancala_env as mancala
import time
import copy
import random
import random_agent
import left_agent
import right_agent
import manual
import minimax_agent as minimax
import sys
import importlib

#----------------
#-----GLOBAL-----
#----------------

runs = 100

#----------------
#---END GLOBAL---
#----------------

def main():
    if len(sys.argv) < 2:
        #if using the minimax agent -> must be player_0
        agent_function = { "player_0": minimax.agent_function, "player_1": manual.agent_function }
        # agent_function = { "player_0": random_agent.agent_function, "player_1": random_agent.agent_function }
        times = { "player_0": 0.0, "player_1": 0.0 }

        env = mancala.env(render_mode="human")
        env.reset(seed=42)

        for agent in env.agent_iter():

            #step() checks for game_over and displays winner results
            #must check here to end loop once over
            if env.GAME_OVER():
                break

            t1 = time.time()
            action = agent_function[agent](env, agent)
            t2 = time.time()
            times[agent] += (t2-t1)

            #if action != None:
            env.step(action)

        # time.sleep(10) # useful for end of game with human render mode
        env.close()

        for agent in times:
            print(f"{agent} took {times[agent]:8.5f} seconds.")
        return
    
    if len(sys.argv) != 3:
        print("Error incorrect amount of arguments")
        print("Usage: python3 play.py _or_ python3 play.py <agent_0> <agent_1>")
        return
    
    else:
        #if using the minimax agent -> must be player_0
        agent_function = { "player_0": importlib.import_module(sys.argv[1]).agent_function, "player_1": importlib.import_module(sys.argv[2]).agent_function }
        # agent_function = { "player_0": random_agent.agent_function, "player_1": random_agent.agent_function }
        times = { "player_0": 0.0, "player_1": 0.0 }
        scores = { "player_0": 0, "player_1": 0 }
        wins = { "player_0": 0, "player_1": 0 }
        ties = 0

        for run in range(runs):
            env = mancala.env(render_mode="human")
            env.reset(seed=42, options=True)

            for agent in env.agent_iter():

                #step() checks for game_over and displays winner results
                #must check here to end loop once over
                if env.GAME_OVER():
                    break

                t1 = time.time()
                action = agent_function[agent](env, agent)
                t2 = time.time()
                times[agent] += (t2-t1)

                #if action != None:
                env.step(action)

            game_total = env.get_final_score()

            scores["player_0"] += game_total["player_0"]
            scores["player_1"] += game_total["player_1"]

            if game_total["player_0"] > game_total["player_1"]:
                wins["player_0"] += 1
            elif game_total["player_0"] < game_total["player_1"]:
                wins["player_1"] += 1
            else:
                ties += 1


            # time.sleep(10) # useful for end of game with human render mode
            env.close()

        print("\n")
        for agent in times:
            print(f"{agent} took {(times[agent]/runs):8.5f} seconds on average.")
        print("\n")
        for agent in scores:
            print(f"{agent} scored {(scores[agent]/runs):8.5f} points on average.")
        print("\n")
        for agent in wins:
            print(f"{agent} won {wins[agent]:8.5f} times.")
        print("\n")
        print("Ties:", ties)
        return  

if __name__ == "__main__":
    if False:
        import cProfile
        cProfile.run('main()')
    else:
        main()
