import sys, re

f2 = file("temp.mak", "wt")
for line in file(sys.argv[1]):
    line = re.sub(r"(?i)\\D\\ai\\Orange\\source\\", r"..\\", line)
    line = re.sub(r"(?i)\.[/\\]obj[/\\]Release", r".\\obj\\Release", line)
    line = re.sub(r"(?i)\.[/\\]obj[/\\]Release[/\\]", r".\\obj\\Release\\", line)
    line = re.sub(r"(?i)\$\(OUTDIR\)/\$\(NULL\)", r"$(OUTDIR)\\$(NULL)", line)
    f2.write(line)


f2.close()
