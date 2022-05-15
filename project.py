# Artificial Intelligence - Final Project Assignment
# Coded by Laura Belizón Merchán and Jorge Lázaro Ruiz

import difflib


def bellman(current_state, actions, possible_states, prev_val) -> int:
    """Returns the next expected value of a state, using the Bellman equations."""
    if current_state == LLL:
        return 0  # LLL is our goal state, and V(goal) = 0

    arg = []  # Here we will store the result for each action

    # This is an implementation of the Bellman equations
    for a in actions:
        x = 0
        x += cost[a]
        for t in possible_states:
            x += T[current_state][a][t] * prev_val[t]
        arg.append(x)

    return round(min(arg), 6)  # Find the minimum and round it to six decimal places


# We are going to define some constants in order to improve readability and to generalize the problem
TRIALS = 8785
ACTIONS = 3
STATES = 8

# FORMAL DEFINITION OF OUR MARKOV DECISION PROCESS <S, A, T, R>
# [S] Possible states of the traffic flux (format: WNE)
LLL = 0  # 0 is our goal state
LLH = 1
LHL = 2
LHH = 3
HLL = 4
HLH = 5
HHL = 6
HHH = 7

# [A] Possible actions the automaton can take (turning a traffic light green)
WEST = 0
NORTH = 1
EAST = 2

# [T] A STATES x ACTIONS x STATES (8 x 3 x 8 in our case) matrix that will store probability of each transition
# Entry T[s][a][s'] stores the value of T(s, a, s')
T = [[[0 for i in range(STATES)] for j in range(ACTIONS)] for k in range(STATES)]

# [R] We will take the cost as the reward function
COST = 20  # We are going to assume the cost of turning any of the traffic lights green is 20 seconds
cost = {WEST: COST, NORTH: COST, EAST: COST}  # Store the cost of each action in a dictionary

states = []
for i in range(0, STATES):
    states.append(i)

actions = []
for i in range(0, ACTIONS):
    actions.append(i)

# Obtaining the probability table for the transitions
print("Reading and processing historical data...")
history = open('Data.csv', 'r')  # Read the file with historical data
Lines = history.readlines()  # Read the data line by line

count = -1  # Counts the lines; starts at -1 to offset the header in the CSV file
possibilities = ["LowLowLow", "LowLowHigh", "LowHighLow", "LowHighHigh",
                 "HighLowLow", "HighLowHigh", "HighHighLow", "HighHighHigh"]  # Strings representing possible states
for line in Lines:
    count += 1  # Add a line to the count

    if count == 0:
        continue  # Skip header line

    split = line.split(';')  # Split the line using the semicolon as a delimiter
    s = split[0] + split[1] + split[2]  # Concatenate to obtain initial state "s"
    a = split[3]  # Action "a"
    t = split[4] + split[5] + split[6]  # Concatenate to obtain new state "s'" (called t from now on)

    # We use the difflib.get_close_matches() function in case the human compiling the data makes a typo
    # This way our AI is more robust against human errors
    s = difflib.get_close_matches(s, possibilities, 1)
    a = difflib.get_close_matches(a, ["W", "N", "E"], 1)
    t = difflib.get_close_matches(t, possibilities, 1)

    # Convert the state "s" to a number
    if s[0] == possibilities[0]:
        s = LLL
    elif s[0] == possibilities[1]:
        s = LLH
    elif s[0] == possibilities[2]:
        s = LHL
    elif s[0] == possibilities[3]:
        s = LHH
    elif s[0] == possibilities[4]:
        s = HLL
    elif s[0] == possibilities[5]:
        s = HLH
    elif s[0] == possibilities[6]:
        s = HHL
    elif s[0] == possibilities[7]:
        s = HHH

    # Convert the action "a" to a number
    if a[0] == "W":
        a = WEST
    elif a[0] == "N":
        a = NORTH
    elif a[0] == "E":
        a = EAST

    # Convert the state "t" to a number
    if t[0] == possibilities[0]:
        t = LLL
    elif t[0] == possibilities[1]:
        t = LLH
    elif t[0] == possibilities[2]:
        t = LHL
    elif t[0] == possibilities[3]:
        t = LHH
    elif t[0] == possibilities[4]:
        t = HLL
    elif t[0] == possibilities[5]:
        t = HLH
    elif t[0] == possibilities[6]:
        t = HHL
    elif t[0] == possibilities[7]:
        t = HHH

    T[s][a][t] += 1 / TRIALS  # Equally likely events

history.close()  # Close the file with historical data
print("Done.")

# Value iteration
print("Performing value iteration...")
value_iteration = [[0 for i in range(STATES)]]  # Every expected value starts at 0

i = 0
iteration = []
for s in states:
    iteration.append(bellman(s, actions, states, value_iteration[i]))
value_iteration.append(iteration)
i += 1  # Iteration 1 has to be done outside the loop

while value_iteration[i] != value_iteration[i - 1]:  # Our criteria for determining when a value converges
    iteration = []
    for s in states:
        iteration.append(bellman(s, actions, states, value_iteration[i]))
    value_iteration.append(iteration)
    i += 1
print("Done.\n-----------------------------------")

# Optimal policies
op = {}  # Dictionary that will store the optimal policy for each state
for s in states:
    if s == 0:
        print("Optimal policy for state", s, "is undefined")

    else:  # Very similar to the code found in the bellman() function
        arg = []
        for a in actions:
            x = 0
            x += cost[a]
            for t in states:
                x += T[s][a][t] * value_iteration[-1][t]
            arg.append(x)
        print(f"Optimal policy for state {s} is action {arg.index(min(arg))}")
        op[s] = arg.index(min(arg))
