import pickle

import flappy
import flappy_no_screen
import numpy as np
import sys

from collections import defaultdict
reward = 1
penalty = -10000
alpha = 0.1         # learning rate
gamma = 1         # discount-Factor
learn = False

if learn:
    Q = defaultdict(lambda: [0, 0])
else:
    Q = None
#Q["S1"] = {1, -1000}


def load_q_data():
    global Q
    with open("./Q_data/Q40000.pickle", "rb") as file:
        Q = defaultdict(lambda: [0, 0], pickle.load(file))


if not learn:
    load_q_data()

#sys.exit()
def params_to_state(params):
    playerVelY = params['playerVelY']
    playery = params['playery']
    #print(params)
    if int(params['upperPipes'][0]['x']) < 40:
        upperPipeX = round(int(params['upperPipes'][1]['x']) / 3) * 3
        upperPipeY = int(params['upperPipes'][1]['y'])
    else:
        upperPipeX = round(int(params['upperPipes'][0]['x']) / 3) * 3
        upperPipeY = int(params['upperPipes'][0]['y'])
    y_diff = round((playery-upperPipeY)/5) * 5
    return str(y_diff) + "_"+str(playerVelY) + "_" +str(upperPipeX) + "_"


counter = 0
score_list = []


def on_game_over(gameInfo):
    global old_action, old_state, counter, score_list
    score_list.append(gameInfo['score'])
    if counter % 1000 == 0:
        print(counter, ":", np.mean(score_list))
        score_list.clear()
    prev_reward = Q[old_state]
    index = None
    if old_action == False:
        index = 0
    else:
        index = 1
    prev_reward[index] = (1 - alpha) * prev_reward[index] + alpha * (penalty + gamma * 0)
    Q[old_state] = prev_reward
    #   print(Q)
    old_state = None
    old_action = None
    if learn:
        save_file()


def save_file():
    global counter
    if counter % 10000 == 0:
        with open(".\Q_data\Q"+str(counter)+".pickle", "wb") as file:
            pickle.dump(dict(Q), file)
    counter += 1


old_state = None     # status im Frame davor
old_action = None


def should_emulate_key_press(params):
    global old_action, old_state
    state = params_to_state(params)
    prev_reward = Q[old_state]
    index = None
    if old_action == False:
        index = 0
    else:
        index = 1
    est_reward = Q[state]
    prev_reward[index] = (1-alpha) * prev_reward[index] + alpha * (reward + gamma * max(est_reward))
    Q[old_state] = prev_reward

    old_state = state

    if est_reward[0] >= est_reward[1]:
        old_action = False
        return False
    else:
        old_action = True
        return True

    #return np.random.choice([False, True],p=[0.9, 0.1])
    #return True


if not learn:
    flappy.main(should_emulate_key_press, on_game_over)
else:
    flappy_no_screen.main(should_emulate_key_press, on_game_over)