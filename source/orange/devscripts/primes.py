#
# Alex was here!
#

import os
os.chdir(os.getenv("ORANGEHOME")+"/source/include")

sieve = [1] * 1000000

for i in xrange(2, 1000):
    if sieve[i]:
        for j in xrange(2*i, 100000, i):
            sieve[j] = 0


outf = file("primes.c", "wt")
inline = 10

outf.write("long primes[] = {")

for i in xrange(2, 60000):
    if sieve[i]:
        if inline == 10:
            outf.write("\n   ")
            inline = 0
        else:
            inline += 1
        outf.write("%6i, " % i)

outf.write("%6i};\n" % 0)
outf.close()
