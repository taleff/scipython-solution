import sys

#P2.5.7a
start = int(sys.argv[1])

sequence = []
while start != 1:
    sequence.append(start)
    if start % 2:
        start = 3 * start + 1
    else:
        start = start // 2
    print(start)
print(1)
