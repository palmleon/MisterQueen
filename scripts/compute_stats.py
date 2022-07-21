from cmath import inf

print("inserisci nome file 1:")
fn = str(input())
f = open(fn)
lines1 = f.readlines()
f.close()
print("inserisci nome file 2:")
fn = str(input())
f = open(fn)
lines2 = f.readlines()
f.close()
min = inf
max = 0
tot = 0
avg = 0
for i in range(len(lines1)):
    if (i%2 == 0):
        tmp = float(lines2[i].split()[-2])/float(lines1[i].split()[-2])
        if (tmp < min):
            min = tmp
        if (tmp > max):
            max = tmp
        tot += tmp
avg = tot/(len(lines1)/2)
f = open(fn[0:-4]+"_speedup.txt", "w")
f.write("min = " + str(min) + "\n")
f.write("max = " + str(max) + "\n")
f.write("avg = " + str(avg) + "\n")