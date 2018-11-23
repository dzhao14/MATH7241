import numpy as np
np.set_printoptions(threshold=np.inf, precision=4)

num_states = 50

ff = open('filtered_commands_{}'.format(num_states), 'r')
lines = ff.readlines()
ff.close()

states = {}

for line in lines:
    command = line[:-1]
    if command not in states:
        states[command] = 1
    else:
        states[command] += 1

freq = [(key, val) for key, val in states.items()]
freq.sort(key=lambda tup: tup[1])

index = {}
for i, t in enumerate(freq):
    key, val = t
    index[key] = num_states-i


P = np.zeros(shape=(num_states, num_states))

prev = lines[0][:-1]
for line in lines[1:]:
    command = line[:-1]
    prev_idx = index[prev] - 1
    cur_idx = index[command] - 1
    P[prev_idx,cur_idx] += 1.0
    prev = command

for i in range(num_states):
    row_sum = sum(P[i,:])
    for j in range(num_states):
        P[i,j] /= (1.0 * row_sum)
        #if P[i,j] < 0.05:
        #    P[i,j] = 0

#np.savetxt("transition_top_{}.csv".format(num_states), P, delimiter=",")

mul = np.matmul(P, P)
for i in range(1000):
    mul = np.matmul(mul, P)

#for i in range(num_states):
#    for j in range(num_states):
#        assert mul[i,j] > 0.0

print(mul[0,:])
