import orange
import orngFSS
import statc
import orngCI

###########################################################################################
##### FUNCTIONS FOR CALCULATING ATTRIBUTE ORDER USING Oblivious decision graphs
###########################################################################################
def replaceAttributes(index1, index2, merged, data):
    attrs = list(data.domain.attributes)
    attrs.remove(data.domain[index1])
    attrs.remove(data.domain[index2])
    return data.select([merged] + attrs + [data.domain.classVar])


def getFunctionalList(data):
    bestQual = -10000000
    bestAttrs = []
    testAttrs = []
    outList = []

    dataShort = orange.Preprocessor_dropMissing(data)
    # remove continuous attributes from data
    disc = []
    for i in range(len(dataShort.domain.attributes)):
        if dataShort.domain.attributes[i].varType == orange.VarTypes.Discrete: disc.append(dataShort.domain.attributes[i].name)
    if disc == []: return []
    discData = dataShort.select(disc + [dataShort.domain.classVar.name])

    remover = orngCI.AttributeRedundanciesRemover(noMinimization = 1)
    newData = remover(discData, weight = 0)

    # ####
    # compute the best attribute combination
    # ####
    for i in range(len(newData.domain.attributes)):
        if newData.domain.attributes[i].varType != orange.VarTypes.Discrete: continue
        testAttrs.append(newData.domain.attributes[i].name)
        for j in range(i+1, len(newData.domain.attributes)):
            if newData.domain.attributes[j].varType != orange.VarTypes.Discrete: continue
            vals, qual = orngCI.FeatureByMinComplexity(newData, [newData.domain.attributes[i], newData.domain.attributes[j]])
            if qual > bestQual:
                bestQual = qual
                bestAttrs = [newData.domain.attributes[i].name, newData.domain.attributes[j].name, vals]

    if bestAttrs == []: return []
    outList.append(bestAttrs[0])
    outList.append(bestAttrs[1])
    newData = replaceAttributes(bestAttrs[0], bestAttrs[1], bestAttrs[2], newData)
    testAttrs.remove(bestAttrs[0])
    testAttrs.remove(bestAttrs[1])
    
    while (testAttrs != []):
        bestQual = -10000000
        for attrName in testAttrs:
            vals, qual = orngCI.FeatureByMinComplexity(newData, [newData.domain[0], attrName])
            if qual > bestQual:
                bestqual = qual
                bestAttrs = [0, attrName, vals]
        newData = replaceAttributes(0, bestAttrs[1], bestAttrs[2], newData)
        outList.append(bestAttrs[1])
        testAttrs.remove(bestAttrs[1])

    outList.reverse()
    # new attributes have "'" at the end of their names. we have to remove that in ored to identify them in the old domain
    for index in range(len(outList)):
        if outList[index][-1] == "'": outList[index] = outList[index][:-1]
    return outList


###########################################################################################
##### FUNCTIONS FOR CALCULATING ATTRIBUTE ORDER USING CORRELATION
###########################################################################################

def insertToSortedList(array, val, names):
    for i in range(len(array)):
        if val > array[i][0]:
            array.insert(i, [val, names])
            return
    array.append([val, names])

# does value exist in array? return index in array if yes and -1 if no
def member(array, value):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j]==value:
                return i
    return -1
        

# insert two attributes to the array in such a way, that it will preserve the existing ordering of the attributes
def insertToCorrList(array, attr1, attr2):
    index1 = member(array, attr1)
    index2 = member(array, attr2)
    if index1 == -1 and index2 == -1:
        array.append([attr1, attr2])
    elif (index1 != -1 and index2 == -1) or (index1 == -1 and index2 != -1):
        if index1 == -1:
            index = index2
            newVal = attr1
            existingVal = attr2
        else:
            index = index1
            newVal = attr2
            existingVal = attr1
            
        # we insert attr2 into existing set at index index1
        pos = array[index].index(existingVal)
        if pos < len(array[index])/2:   array[index].insert(0, newVal)
        else:                           array[index].append(newVal)
    else:
        # we merge the two lists in one
        if index1 == index2: return
        array[index1].extend(array[index2])
        array.remove(array[index2])
    

# create such a list of attributes, that attributes with high correlation lie together
def getCorrelationList(data):
    # create ordinary list of data values    
    dataList = []
    dataNames = []
    for index in range(len(data.domain)):
        if data.domain[index].varType != orange.VarTypes.Continuous: continue
        temp = []
        for i in range(len(data)):
            temp.append(data[i][index])
        dataList.append(temp)
        dataNames.append(data.domain[index].name)

    # compute the correlations between attributes
    correlations = []
    for i in range(len(dataNames)):
        for j in range(i+1, len(dataNames)):
            val, prob = statc.pearsonr(dataList[i], dataList[j])
            insertToSortedList(correlations, abs(val), [i,j])
            #print "correlation between %s and %s is %f" % (dataNames[i], dataNames[j], val)

    i=0
    mergedCorrs = []
    while i < len(correlations) and correlations[i][0] > 0.1:
        insertToCorrList(mergedCorrs, correlations[i][1][0], correlations[i][1][1])
        i+=1

    hiddenList = []
    while i < len(correlations):
        if member(mergedCorrs, correlations[i][1][0]) == -1:
            hiddenList.append(dataNames[correlations[i][1][0]])
        if member(mergedCorrs, correlations[i][1][1]) == -1:
            hiddenList.append(dataNames[correlations[i][1][1]])
        i+=1

    shownList = []
    for i in range(len(mergedCorrs)):
        for j in range(len(mergedCorrs[i])):
            shownList.append(dataNames[mergedCorrs[i][j]])

    if len(dataNames) == 1: shownList += dataNames
    return (shownList, hiddenList)


##############################################
# SELECT ATTRIBUTES ##########################
##############################################
def selectAttributes(data, attrContOrder, attrDiscOrder):
	shown = []; hidden = []	# initialize outputs

	## both are RELIEF
	if attrContOrder == "RelieF" and attrDiscOrder == "RelieF":
		newAttrs = orngFSS.attMeasure(data, orange.MeasureAttribute_relief(k=20, m=50))
		for item in newAttrs:
			if float(item[1]) > 0.01:   shown.append(item[0])
			else:                       hidden.append(item[0])
		return (shown, hidden)

	## both are NONE
	elif attrContOrder == "None" and attrDiscOrder == "None":
		for item in data.domain.attributes:    shown.append(item.name)
		return (shown, hidden)

	###############################
	# sort continuous attributes
	if attrContOrder == "None":
		for item in data.domain:
			if item.varType == orange.VarTypes.Continuous: shown.append(item.name)
	elif attrContOrder == "RelieF":
		newAttrs = orngFSS.attMeasure(data, orange.MeasureAttribute_relief(k=20, m=50))
		for item in newAttrs:
			if data.domain[item[0]].varType != orange.VarTypes.Continuous: continue
			if float(item[1]) > 0.01:   shown.append(item[0])
			else:                       hidden.append(item[0])
	elif attrContOrder == "Correlation":
		(shown, hidden) = getCorrelationList(data)    # get the list of continuous attributes sorted by using correlation
	else:
		print "Incorrect value for attribute order"

	################################
	# sort discrete attributes
	if attrDiscOrder == "None":
		for item in data.domain.attributes:
			if item.varType == orange.VarTypes.Discrete: shown.append(item.name)
	elif attrDiscOrder == "RelieF":
		newAttrs = orngFSS.attMeasure(data, orange.MeasureAttribute_relief(k=20, m=50))
		for item in newAttrs:
			if data.domain[item[0]].varType != orange.VarTypes.Discrete: continue
			if item[0] == data.domain.classVar.name: continue
			if float(item[1]) > 0.01:   shown.append(item[0])
			else:                       hidden.append(item[0])
	elif attrDiscOrder == "GainRatio" or attrDiscOrder == "Gini":
		if attrDiscOrder == "GainRatio":   measure = orange.MeasureAttribute_gainRatio()
		else:                                   measure = orange.MeasureAttribute_gini()
		if data.domain.classVar.varType != orange.VarTypes.Discrete:
			measure = orange.MeasureAttribute_relief(k=20, m=50)

		# create new table with only discrete attributes
		attrs = []
		for attr in data.domain.attributes:
			if attr.varType == orange.VarTypes.Discrete: attrs.append(attr)
		attrs.append(data.domain.classVar)
		dataNew = data.select(attrs)
		newAttrs = orngFSS.attMeasure(dataNew, measure)
		for item in newAttrs:
				shown.append(item[0])

	elif attrDiscOrder == "Oblivious decision graphs":
            shown.append(data.domain.classVar.name)
            list = getFunctionalList(data)
            for item in list:
                shown.append(item)
            for attr in data.domain.attributes:
                if attr.name not in shown and attr.varType == orange.VarTypes.Discrete:
                    hidden.append(attr.name)
	else:
		print "Incorrect value for attribute order"

	#################################
	# if class attribute hasn't been added yet, we add it
	if data.domain.classVar.name not in shown and data.domain.classVar.name not in hidden:
		shown.append(data.domain.classVar.name)


	return (shown, hidden)