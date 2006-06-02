import sys

f2 = file("temp.mak", "wt")
for line in file(sys.argv[1]):
    f2.write(line.replace("\\D\\ai\\Orange\\source\\", "..\\"))
f2.close()
