# Description: Shows how to use domain depots
# Category:    basic classes
# Classes:     Domain, DomainDepot
# Uses:        
# Referenced:  DomainDepot.htm

# xtest: RANDOM

import orange

de = orange.DomainDepot()

names = ['mS#name', 'C#age', 'D#gender', 'D#race', 'cC#total', 'mS#SSN']
domain, metaIDs, isNew = de.prepareDomain(names)
domainx = domain
print "Names: ", names
print "IDs of meta attributes: ", metaIDs
print "Is new? ", bool(isNew)
print

names = ['mS#SSN', 'mS#name', 'C#age', 'D#gender', 'D#race', 'cC#total']
domain, metaIDs, isNew = de.prepareDomain(names)
print "Names: ", names
print "IDs of meta attributes: ", metaIDs
print "Is new? ", bool(isNew)
print

names = ['mS#SSN', 'D#gender', 'C#race', 'cC#total']
domain2, metaIDs2, isNew2 = de.prepareDomain(names, domain.attributes, domain.getmetas())
print "Names: ", names
print "IDs of meta attributes: ", metaIDs2
print "Is new? ", bool(isNew2)
for name in names:
    undname = name.split("#")[1]
    print "Is '%s' same?" % undname, domain[undname] == domain2[undname]
print

names = ['mS#SSN', 'C#race', 'D#gender', 'cC#total']
domain2, metaIDs2, isNew2 = de.prepareDomain(names, domain.variables, domain.getmetas())
print "Names: ", names
print "IDs of meta attributes: ", metaIDs2
print "Is new? ", bool(isNew2)
for name in names:
    undname = name.split("#")[1]
    print "Is '%s' same?" % undname, domain[undname] == domain2[undname]
print

names = ['mS#name', 'C#age', 'D#gender', 'D#race', 'cC#total']
domain, metaIDs, isNew = de.prepareDomain(names)
print "Names: ", names
print "Is new? ", bool(isNew)
print

names = ['mS#SSN', 'mS#name', 'D#race', 'C#age', 'D#gender', 'cC#total']
domain, metaIDs, isNew = de.prepareDomain(names)
print "Names: ", names
print "Is new? ", bool(isNew)
print

names = ['mS#SSN', 'mS#name', 'C#age', 'C#gender', 'D#race', 'cC#total']
domain, metaIDs, isNew = de.prepareDomain(names)
print "Names: ", names
print "IDs of meta attributes: ", metaIDs
print "Is new? ", bool(isNew)
print


names = ['D#v%i' % i for i in range(5)]
domain1, mid, isNew = de.prepareDomain(names, None, None, 1)
domain2, mid, isNew = de.prepareDomain(names, None, None)
print "I constructed two same domains, but without storing the first."
print "Is the second new? ", bool(isNew)
print

domain3, mid, isNew = de.prepareDomain(names, None, None, 0, 1)
print "I've stored the second and constructed the third without looking for old domains."
print "Is the third new? ", bool(isNew)
print

domain4, mid, isNew = de.prepareDomain(names, None, None)
print "Finally, I've constructed the fourth domain, without masking anything."
print "Is it new? ", bool(isNew)
print

print "Which one is it equal to?",
for n, d in [("first", domain1), ("second", domain2), ("third", domain3)]:
    if d == domain4:
        print n,
print

for d in [domain1, domain2, domain3, domain4]:
    print de.checkDomain(d, names)
names.append("D#vX")
print de.checkDomain(domain1, names)