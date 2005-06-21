from copy import copy

differentClassPermutationsDict = {}
projectionListDict = {}

# compute permutations of elements in alist
def permutations(alist, blist=[], retList = []):
    if not len(alist): return [copy(blist)]
    retList = []
    for i in range(len(alist)):
        blist.append(alist.pop(i))
        retList += permutations(alist, blist)
        alist.insert(i, blist.pop())
    return retList

def combinations(items, count):
    if count > len(items): return []
    answer = []
    indices = range(count)
    indices[-1] = indices[-1] - 1
    while 1:
        limit = len(items) - 1
        i = count - 1
        while i >= 0 and indices[i] == limit:
            i = i - 1
            limit = limit - 1
        if i < 0: break

        val = indices[i]
        for i in xrange( i, count ):
            val = val + 1
            indices[i] = val
        temp = []
        for i in indices:
            temp.append( items[i] )
        answer.append( temp )
    return answer

# input: array where items are arrays, e.g.: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
# returns: array, where some items are removed. it rotates items in each array and checks if the item is already in the array. it also tries to flip the items ([::-1])
# for the above case, it would return [[1,2,3]]
# used for radviz
"""
def removeRotationAndMirroringDuplicates(arr):
    final = []
    projDict = {}
    print "start...", arr
    a2 = removeRotationAndMirroringDuplicates2(copy(arr))
    for a in arr:
        #print a
        if type(a[0]) == int: key = tuple([a[i]-a[i+1] for i in range(-1, len(a)-1)])
        else:                 key = tuple([(a[i][0]-a[i+1][0], a[i][1]-a[i+1][1]) for i in range(-1, len(a)-1)])
        if projDict.has_key(key): continue
        found = 0
        kkey = copy(key)
        for i in range(len(kkey)):
            kkey = tuple(list(kkey[1:]) + [kkey[0]])
            #kkey = kkey[1:] + [kkey[0]]
            if projDict.has_key(kkey) or projDict.has_key(kkey[::-1]):
                found = 1
                if a in a2: print "not found:  ", a, "\nsame as in: ",
                if projDict.has_key(kkey): print projDict[kkey], "\n", key, "\n", kkey
                else: print projDict[kkey[::-1]], "\n", key, "\n", kkey
                break
        if not found: projDict[key] = a
    print "end...", len(projDict.values()), len(a2)
    return projDict.values()
"""
def removeRotationDuplicates(arr, removeFlipDuplicates = 1):
    final = []
    projDict = {}
    for a in arr:
        if type(a[0]) == int:
            key = tuple([a[i]-a[i+1] for i in range(-1, len(a)-1)])
            rkey = tuple([a[i+1]-a[i] for i in range(-1, len(a)-1)])
        else:
            key = tuple([(a[i][0]-a[i+1][0], a[i][1]-a[i+1][1]) for i in range(-1, len(a)-1)])
            rkey = tuple([(a[i+1][0]-a[i][0], a[i+1][1]-a[i][1]) for i in range(-1, len(a)-1)])

        if projDict.has_key(key) or projDict.has_key(rkey): continue
        
        found = 0
        kkey = copy(key)
        for i in range(len(kkey)):
            kkey = tuple(list(kkey[1:]) + [kkey[0]])
            if projDict.has_key(kkey):
                found = 1
                break

        # try also if there is a flipped duplicate
        if not found and removeFlipDuplicates:
            for i in range(len(rkey)):
                rkey = tuple(list(rkey[1:]) + [rkey[0]])
                if projDict.has_key(rkey):
                    found = 1
                    break
        if not found: projDict[key] = a
    return projDict.values()
        
"""
# input: array where items are arrays, e.g.: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
# returns: array, where some items are removed. it rotates items in each array and checks if the item is already in the array. it also tries to flip the items ([::-1])
# for the above case, it would return [[1,2,3]]
# used for radviz
def removeRotationAndMirroringDuplicates2(arr):
    final = []

    while arr != []:
        perm = arr.pop()
        if perm[::-1] in arr: continue
        found = 0
        for i in range(len(perm)):
            perm = perm[1:] + [perm[0]]
            if perm in arr or perm[::-1] in arr:
                found = 1
                break
        if not found: final.append(perm)
    return final

# same as above, just that it doesn't try to flip the image. used for polyviz
def removeRotationDuplicates(arr):
    final = []
    while arr != []:
        perm = arr.pop()
        found = 0
        for i in range(len(perm)):
            perm = perm[1:] + [perm[0]]
            if perm in arr:
                found = 1
                break
        if not found: final.append(perm)
    return final
"""

# create possible combinations with the given set of numbers in arr
def createMixCombinations(arrs, removeFlipDuplicates):
    projs = [[]]
    for i in range(len(arrs)):
        projs = addProjs(projs, arrs[i], i)
    return removeRotationDuplicates(projs, removeFlipDuplicates)


def addProjs(projs, count, i):
    ret = []
    perms = permutations(range(count))
    for perm in perms:
        c = copy(projs)
        add = [(i, p) for p in perm]
        c = [p + add for p in c]
        ret += c
    return ret

# get a list of possible projections if we have numClasses and want to use maxProjLen attributes
# removeFlipDuplicates tries to flip the attributes in the projection and removes it if the projection already exists
# removeFlipDuplicates = 1 for radviz and =0 for polyviz
def createProjections(numClasses, maxProjLen, removeFlipDuplicates = 1):
    if projectionListDict.has_key((numClasses, maxProjLen, removeFlipDuplicates)): return projectionListDict[(numClasses, maxProjLen, removeFlipDuplicates)]

    # create array of arrays of lengths, e.g. [3,3,2] that will tell that we want 3 attrs from the 1st class, 3 from 2nd and 2 from 3rd
    if maxProjLen % numClasses != 0:
        cs = combinations(range(numClasses), maxProjLen % numClasses)
        equal = [int(maxProjLen / numClasses) for i in range(numClasses)]
        lens = [copy(equal) for comb in cs]     # make array of arrays
        for i, comb in enumerate(cs):
            for val in comb: lens[i][val] += 1
    else:
        lens = [[int(maxProjLen / numClasses) for i in range(numClasses)]]
        
    combs = []
    for l in lens:
        tempCombs = createMixCombinations(l, removeFlipDuplicates)
        combs += tempCombs

    if differentClassPermutationsDict.has_key((numClasses, removeFlipDuplicates)):
        perms = differentClassPermutationsDict[(numClasses, removeFlipDuplicates)]
    else:
        perms = permutations(range(numClasses))
        perms = removeRotationDuplicates(perms, removeFlipDuplicates)
        differentClassPermutationsDict[(numClasses, removeFlipDuplicates)] = perms
            
    final = []
    for perm in perms:
        final += [[(perm[i], j) for (i,j) in comb] for comb in combs]

    projectionListDict[(numClasses, maxProjLen, removeFlipDuplicates)] = final
    return final



if __name__== "__main__": 
    #a = []
    #a = permutations([1,2,3], [])
    l=createProjections(2, 4)