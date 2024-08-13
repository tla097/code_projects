#!/usr/bin/env python3
import map


def transitions(people):

    transitions = {
        "RIGHT": transition_states("RIGHT", 1, 0, people),
        "LEFT": transition_states("LEFT", -1, 0, people),
        "UP": transition_states("UP", 0, 1, people),
        "DOWN": transition_states("DOWN", 0, -1, people),
        "TERMINAL": transition_states("TERMINAL", 0, 0, people)
    }
    return transitions


def transition_states(direction, x, y, people):
    """
    calculates transition matrix given probabilities

    :param direction: direction of movement
    :param x: what to add to the x axis to go in that direction
    :param y: what to add to the y axis to go in that direction
    :return: transition probabilities dictionary
    """
    # add heatmap uncertainty

    states = map.states()
    possible_transitions = map.possible_transitions()
    transition_dict = {}

    uncertainty = 1
    for state in states:
        transition_dict[state] = {}
        for next_state in states:
            if state == next_state:                           # set them all to 1 initially and then if it turns out you
                transition_dict[state][next_state] = 1     # <- don't have to stay there then it gets updated later
            elif direction in possible_transitions[state] \
                    and states[state][0] + x == states[next_state][0] \
                    and states[state][1] + y == states[next_state][1]:  # if you can go in the direction
                uncertainty = (people.count(int(''.join(filter(str.isdigit, next_state))))) * 0.29 + 0.1
                # calculates the number of people in the cell and then works out the probability of transition
                transition_dict[state][next_state] = 1 - uncertainty
            else:
                transition_dict[state][next_state] = 0
        transition_dict[state].update({state: uncertainty})

    return transition_dict




# We don't use these
def calc_state_string(sx, sy):
    return "s" + str(sx + sy * 11)


def state_int(x, y):
    return x + y * 11


def generate_states():
    grid_map = map.grid_map()
    possible_states = {}
    state_no = "s"
    i = 0
    for row in range(11):
        for col in range(9):
            if grid_map[row][col] == 0:
                possible_states[str(state_no) + str(i)] = (col, row)
                i += 1
    return possible_states
