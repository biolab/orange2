import orngStat

CAs = orngStat.CA(res)
APs = orngStat.AP(res)
Briers = orngStat.BrierScore(res)
ISs = orngStat.IS(res)

print
print "method\tCA\tAP\tBrier\tIS"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f\t%5.3f\t%6.3f" % (learners[l].name, CAs[l], APs[l], Briers[l], ISs[l])