
f = open('lotsofcommands', 'r')
lines = f.readlines()
f.close()

states = {}
prev = ""
ignore = ["-", "`", ";", "|"]
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
        if prev not in states:
            states[prev] = 1
        else:
            states[prev] += 1
        prev = ""

    prev += command

num_states = 50

freq = [val for key, val in states.items()]
freq = sorted(freq)

filtered_states = {}

print(freq[-num_states:])
s = 0
for key, val in states.items():
    if val in freq[-num_states:]:
        filtered_states[key] = val
        s += val
print (s)


ff = open('lotsofcommands', 'r')
lines = ff.readlines()
ww = open('filtered_commands_{}'.format(num_states), 'w+')

prev = ""
ignore = ["-", "`", ";", "|"]
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
        if prev in filtered_states:
            ww.write(prev + "\n")
        prev = ""

    prev += command

ww.close()

