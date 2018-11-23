import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

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

x = [i+1 for i in range(len(freq))]
y = [i[1] for i in reversed(freq)]

#trace = go.Scatter(x=x, y=y)
#offline.plot([trace], image='png', filename = "emperical_distribution")

index = {}
for i, t in enumerate(freq):
    key, val = t
    index[key] = num_states-i

size = 100
x = [i+1 for i in range(size)]
y = []

for line in lines[:size]:
    command = line[:-1]
    idx = index[command]
    y.append(idx)

trace = go.Scatter(x=x, y=y, mode='markers')
offline.plot([trace], image='png', filename = "emperical_timeseries_{}_states_{}_steps".format(num_states, size))

#P = np.zeros(shape=(num_states, num_states))
#
#prev = lines[0][:-1]
#for line in lines[1:]:
#    command = line[:-1]
#    prev_idx = index[prev] - 1
#    cur_idx = index[command] - 1
#    P[prev_idx,cur_idx] += 1.0
#    prev = command
#
#for i in range(num_states):
#    row_sum = sum(P[i,:])
#    for j in range(num_states):
#        P[i,j] /= (1.0 * row_sum)
#        #if P[i,j] < 0.05:
#        #    P[i,j] = 0


