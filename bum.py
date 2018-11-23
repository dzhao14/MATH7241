import argparse
import numpy as np
import sys
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

from functools import reduce

import ipdb


def parse_data(lines):
    prev = ""
    ignore = ["-", "`", ";", "|"]
    clean_output = []
    for line in lines:
        command = line[:-1]
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
            prev = ""
            
        prev += command

    return clean_output


def parse_all_data(lines):
    contents = list(map(lambda x: parse_data(x), lines)) 
    states = {}
    for file_content in contents:
        for command in file_content:
            if command in states:
                states[command] += 1
            else:
                states[command] = 1
    return contents, states


def create_transition_matrix(index, num_states, filtered_commands):
    commands = reduce(lambda x,y : x + y, filtered_commands, [])
    P = np.zeros(shape=(num_states, num_states))
    prev = commands[0]
    for command in commands[1:]:
        prev_idx = index[prev] - 1
        cur_idx = index[command] - 1
        P[prev_idx, cur_idx] += 1.0
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


def plot_time_series(xlen, filtered_commands, num_states, index, freq):
    """
    ets is short for emperical time series
    """
    colors = [
            "0,0,0",
            "0,0,255",
            "0,255,0",
            "0,255,255",
            "255,0,0",
            "255,0,255",
            "255,255,0",
            "200,200,200",
            "125, 125, 125",
            ]
    prev = 0
    i=0
    data = []
    layout = go.Layout(
            yaxis=dict(
                tickvals = [i+1 for i in range(num_states)],
                ticktext = [i[0] for i in reversed(freq)]
                ),
            )

    for file_commands in filtered_commands:
        x = [i+1 for i in range(prev, prev + len(file_commands))]
        y = []
        for command in file_commands:
            idx = index[command]
            y.append(idx)
        trace = go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker= dict(color='rgb({})'.format(colors[i])),
                )
        data.append(trace)
        prev += len(file_commands)
        i += 1

    ipdb.set_trace() 
    fig = go.Figure(data=data, layout=layout)
    offline.plot(
            fig,
            image='png',
            filename = "ets_{}_states_{}_steps".format(num_states, xlen)
            )


def main(args):
    num_states = int(args.states)
    xlen = int(args.timelength) 
    input_files = [
            "user0",
            "user1",
            "user2",
            "user3",
            "user4",
            "user5",
            "user6",
            "user7",
            "user8",
            ]
    all_unfiltered_content = []
    for input_file in input_files:
        f = open(input_file)
        content = f.readlines()
        f.close()
        all_unfiltered_content.append(content)

    cleaned_inputs, all_freq = parse_all_data(all_unfiltered_content)
    freq = sorted([val for key, val in all_freq.items()])
    filtered_freq = {}
    for key, val in all_freq.items():
        if val in freq[-num_states:]:
            filtered_freq[key] = val

    filtered_commands = []
    for clean_file in cleaned_inputs:
        out = []
        for command in clean_file:
            if command in filtered_freq:
                out.append(command)
        filtered_commands.append(out)

    freq = [(key, val) for key, val in filtered_freq.items()]
    freq.sort(key=lambda tup: tup[1])

    index = {}
    for i, t in enumerate(freq):
        key, val = t
        index[key] = num_states-i

    P = create_transition_matrix(index, num_states, filtered_commands)
    np.savetxt("transition_top_{}.csv".format(num_states), P, delimiter=",")

    if args.ets:
        plot_time_series(xlen, filtered_commands, num_states, index, freq)
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

