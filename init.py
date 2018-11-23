

f = open('lotsofcommands')
lines = f.readlines()

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


#for key, item in states.items():
#    print(key)

freq = [val for key, val in states.items()]
freq = sorted(freq)

print("Number of unique commands: {}".format(len(states)))
print("Total number of commands used: {}".format(sum(freq)))
print("top 50 frequencies: {}".format(freq[-50:]))
print("sum of top 50 frequencies: {}".format(sum(freq[-50:])))
s = 0
c = 0
for key,val in states.items():
    if val > 10:
        s += val
        c += 1
print (s)
print (c)


