import argparse
import numpy as np
import sys
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

import ipdb


def parse_data(lines):
    states = {}
    prev = ""
    ignore = ["-", "`", ";", "|"]
    clean_output = []
    cutoffs = []
    counter = 0
    for line in lines:
        command = line[:-1]
        if command == "delimiter":
            cutoffs.append(counter)
        if len(command) == 0:
            continue

        if command[0] in ignore:
            continue

        if command[0] == "<" and "GENSYM" not in command:
            prev += command 
            continue
            
        if command == "**EOF**" or command == "**SOF**":
            continue

        if prev != "":
            clean_output.append(prev)
            if prev not in states:
                states[prev] = 1
            else:
                states[prev] += 1
            counter += 1
            prev = ""

        prev += command

    print(cutoffs)
    return states, clean_output, cutoffs


def create_transition_matrix(freq, index, num_states, filtered_commands):
    P = np.zeros(shape=(num_states, num_states))
    prev = filtered_commands[0]
    for command in filtered_commands[1:]:
        prev_idx = index[prev] - 1
        cur_idx = index[command] - 1
        P[prev_idx,cur_idx] += 1.0
        prev = command

    for i in range(num_states):
        row_sum = sum(P[i,:])
        for j in range(num_states):
            P[i,j] /= (1.0 * row_sum)
    
    return P


def plot_state_distribution(freq):
    x = [i+1 for i in range(len(freq))]
    y = [i[1] for i in reversed(freq)]
    trace = go.Scatter(x=x, y=y)
    offline.plot([trace], image='png', filename = "emperical_distribution")


def plot_time_series(xlen, filtered_commands, num_states, index, cutoffs):
    """
    ets is short for emperical time series
    """
    if xlen <= cutoffs[0]:
        x = [i+1 for i in range(xlen)]
        y = []
        
        for command in filtered_commands[:xlen]:
            idx = index[command]
            y.append(idx)

        trace = go.Scatter(x=x, y=y, mode='markers')
        offline.plot(
                [trace],
                image='png',
                filename = "ets_{}_states_{}_steps".format(num_states, xlen))
    else:
        colors = [
                "0,0,0",
                "0,0,255",
                "0,255,0",
                "0,255,255",
                "255,0,0",
                "255,0,255",
                "255,255,0",
                "255,255,255",
                "125, 125, 125",
                ]
        i = 0
        data = []
        cutoffs.append(0)
        while xlen > 0 and i < 9:
            x = [i+1 for i in range(cutoffs[i-1], cutoffs[i])]
            y = []
            xlen -= cutoffs[i]
            for command in filtered_commands[cutoffs[i-1]:cutoffs[i]+1]:
                idx = index[command]
                y.append(idx)
            trace = go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker= dict(color='rgb({})'.format(colors[i])),
                    )
            data.append(trace)
            i += 1
            if i >= 9:
                break
        
        ipdb.set_trace()
        offline.plot(
                data,
                image='png',
                filename = "ets_{}_states_{}_steps".format(num_states, xlen)
                )

def main(args):
    num_states = int(args.states)
    xlen = int(args.timelength) 
    f = open('delimitedcommands', 'r')
    lines = f.readlines()
    f.close()

    states, clean_output, cutoffs = parse_data(lines)
    freq = sorted([val for key, val in states.items()])
    filtered_states = {}
    for key, val in states.items():
        if val in freq[-num_states:]:
            filtered_states[key] = val

    filtered_commands = []
    for line in clean_output:
        if line in filtered_states:
            filtered_commands.append(line)

    freq = [(key, val) for key, val in filtered_states.items()]
    freq.sort(key=lambda tup: tup[1])

    index = {}
    for i, t in enumerate(freq):
        key, val = t
        index[key] = num_states-i

    P = create_transition_matrix(freq, index, num_states, filtered_commands)
    np.savetxt("transition_top_{}.csv".format(num_states), P, delimiter=",")

    if args.ets:
        plot_time_series(xlen, filtered_commands, num_states, index, cutoffs)
    if args.sd:
        plot_state_distribution(freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("states")
    parser.add_argument("timelength")
    parser.add_argument(
            "-ets",
            action="store_true",
            help="plots the chain for <timelength> time steps"
            )
    parser.add_argument(
            "-sd",
            action="store_true",
            help="plots the state distribution"
            )

    args = parser.parse_args()

    main(args)

