import argparse
import math
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import sys
import random

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
    
    np.savetxt("transition_top_{}.csv".format(num_states), P, delimiter=",")
    return P


def compute_stationary_distribution(P, num_states, freq):
    p2 = np.matmul(P,P)
    for i in range(1000):
        p2 = np.matmul(p2,P)

    for i in range(num_states):
        for j in range(num_states):
            assert p2[i,j] > 0.0

    return p2[0,:]


def plot_statdist_occupation_freq(P, sd, num_states, freq):
    count = sum([i[1] for i in freq])

    layout = go.Layout(
            title="Stationary Distribution vs. Occupation frequency Distribution",
            xaxis=dict(
                title="Command",
                tickvals = [i+1 for i in range(num_states)],
                ticktext = [i[0] for i in reversed(freq)],
                ),
            )
    trace0 = go.Scatter(
            x=[i+1 for i in range(num_states)],
            y=sd,
            mode='markers',
            name="stationary distribution",
            )
    trace1 = go.Scatter(
            x=[i+1 for i in range(num_states)],
            y = [i[1]/count for i in reversed(freq)],
            name="emperical occupation freq",
            )
    fig = go.Figure(data=[trace0, trace1], layout=layout)
    offline.plot(fig, image='png', filename = "emperical_vs_stat_distribution")


def plot_state_distribution(freq, num_states):
    x = [i+1 for i in range(len(freq))]
    tot = sum([i[1] for i in freq])
    y = [i[1]/tot for i in reversed(freq)]
    layout = go.Layout(
            title="Command Occupation Frequency Distribution",
            xaxis=dict(
                tickvals = [i+1 for i in range(num_states)],
                ticktext = [i[0] for i in reversed(freq)],
                title="command",
                ),
            yaxis=dict(
                title="percent of occurences",
                ),
            )
    trace = go.Scatter(x=x, y=y)
    fig = go.Figure(data=[trace], layout=layout)
    offline.plot(
            fig,
            image='png',
            filename = "occupation_frequency_{}".format(num_states),
            )

 
def simulate_chain(P, xlen, num_states, freq, filtered_commands, index):
    commands = reduce(lambda x,y : x + y, filtered_commands, [])
    run = []
    state = random.randint(0, num_states-1)
    run.append(state+1)
    for i in range(xlen):
        transition_row = P[state, :]
        coin_flip = random.random()
        s = 0
        for next_state, j in enumerate(transition_row):
            if coin_flip >= s and coin_flip <= s+j:
                state = next_state
                run.append(state+1)
                break
            else:
                s += j
    
    layout = go.Layout(
            title="Simulated Chain",
            yaxis=dict(
                tickvals = [i+1 for i in range(num_states)],
                ticktext = [i[0] for i in reversed(freq)],
                title="Command",
                ),
            xaxis=dict(
                title="Timesteps",
                ),
            )
    simulated = go.Scatter(
            x=[i for i in range(xlen)],
            y=run,
            mode="lines+markers",
            name="simulated",
            line=dict(width=1),
            )
    y = []
    for command in commands[:xlen]:
        indx = index[command]
        y.append(indx)
    emperical = go.Scatter(
            x=[i for i in range(xlen)],
            y=y,
            mode="lines+markers",
            name="emperical",
            line=dict(width=1),
            )
    fig = go.Figure(data=[simulated,emperical], layout=layout)
    offline.plot(
            fig,
            image='png',
            filename="simulated_chain_{}".format(num_states)
            )


def calculate_mixing_time(P, num_states, freq, filtered_commands, index, sd):
    commands = reduce(lambda x,y : x + y, filtered_commands, [])

    mixing_var = []
    run = []
    counts = {i+1:0 for i in range(num_states)}
    for command in commands:
        ind = index[command]
        counts[ind] += 1
        tot = sum([val for key, val in counts.items()])
        dist_so_far = np.array([val/tot for key, val in counts.items()])

        variance = np.linalg.norm(sd - dist_so_far, ord=2)
        mixing_var.append(variance)

    layout = go.Layout(
            title="Total variantion distance from empirical distribution to stationary distribution",
            xaxis=dict(title ="steps"),
            )
    trace = go.Scatter(
            x=[i+1 for i in range(len(commands))],
            y=mixing_var,
            mode="lines+markers",
            name="distance from stat. dist.",
            )
    fig = go.Figure(data=[trace], layout=layout)
    offline.plot(
            fig,
            image='png',
            filename="mixing_time_variation_{}".format(num_states)
            )


def plot_entire_time_series(filtered_commands, num_states, index, freq):
    """
    ets is short for emperical time series
    """
    colors = [
            "0,0,255",
            "0,0,0",
            "0,255,0",
            "0,200,200",
            "255,0,0",
            "255,0,255",
            "200,200,0",
            "150,150,150",
            "75, 75, 75",
            ]
    prev = 0
    i=0
    data = []
    layout = go.Layout(
            title="Timeseries of user terminal commands",
            yaxis=dict(
                tickvals = [i+1 for i in range(num_states)],
                ticktext = [i[0] for i in reversed(freq)],
                title="Command",
                ),
            xaxis=dict(
                title="Time",
                )
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
                marker= dict(
                    color='rgb({})'.format(colors[i]),
                    size=1.5,
                    ),
                )
        data.append(trace)
        prev += len(file_commands)
        i += 1

    fig = go.Figure(data=data, layout=layout)
    offline.plot(
            fig,
            image='png',
            filename = "ets_{}_states".format(num_states)
            )


def plot_portion_timeseries(filtered_commands, num_states, index, freq, xlen):
    commands = reduce(lambda x,y : x + y, filtered_commands, [])
    x = [i+1 for i in range(xlen)]
    y = [index[command] for command in commands[:xlen]]
        
    layout = go.Layout(
            title="Timeseries of user terminal commands",
            yaxis=dict(
                tickvals = [i+1 for i in range(num_states)],
                ticktext = [i[0] for i in reversed(freq)],
                title="Command",
                ),
            xaxis=dict(
                title="Timesteps",
                )
            )
    trace = go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            )
    fig = go.Figure(data=[trace], layout=layout)
    offline.plot(
            fig,
            image='png',
            filename = "segement_of_ets".format(num_states),
            )


def main(args):
    num_states = int(args.states)
    xlen = int(args.timelength) 
    input_files = ["user8"]
    all_unfiltered_content = []
    for input_file in input_files:
        f = open("data/{}".format(input_file))
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
    sd = compute_stationary_distribution(P, num_states, freq)

    if args.csd:
        plot_statdist_occupation_freq(P, sd, num_states, freq)
    if args.sc:
        simulate_chain(P, xlen, num_states, freq, filtered_commands, index)
    if args.mt:
        calculate_mixing_time(P, num_states, freq, filtered_commands, index, sd)
    if args.ets:
        plot_entire_time_series(filtered_commands, num_states, index, freq)
    if args.pts:
        plot_portion_timeseries(filtered_commands, num_states, index, freq, xlen)
    if args.of:
        plot_state_distribution(freq, num_states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("states")
    parser.add_argument("timelength")
    parser.add_argument(
            "-ets",
            action="store_true",
            help="plots the chain for all time steps",
            )
    parser.add_argument(
            "-pts",
            action="store_true",
            help="plots the chain for <timelength> time steps",
            )
    parser.add_argument(
            "-csd",
            action="store_true",
            help="plots the state distribution",
            )
    parser.add_argument(
            "-mt",
            action="store_true",
            help="plots the time it takes for the chain to mix",
            )
    parser.add_argument(
            "-sc",
            action="store_true",
            help="plots a simulation of the chain",
            )
    parser.add_argument(
            "-of",
            action="store_true",
            help="plots the occupation frequency distribution",
            )


    args = parser.parse_args()

    main(args)

