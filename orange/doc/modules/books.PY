import orange
import orngText
import orngMDS

data = orange.ExampleTable("bookexcerpts")
datain = data

# process text, obtain TFIDF vectors
p = orngText.Preprocess(language="en")
data = p.removeStopwordsFromExampleTable(data, 0)
data = p.lemmatizeExampleTable(data, 0)
data = orngText.bagOfWords(data)
bof = data
data = orngText.PreprocessorConstructor_tfidf()(data)(data)
data = orngText.Preprocessor_norm()(data, 2)

# compute distance as 1/cos(fi)
distance = orngText.cos(data, distance = 1)

# use MDS for visualization
print "running MDS"
mds=orngMDS.MDS(distance)
mds.run(100)

from pylab import *
colors = ["red", "yellow", "blue"]

points = []
for (i,d) in enumerate(data):
   points.append((mds.points[i][0], mds.points[i][1], d.getclass()))
for c in range(len(data.domain.classVar.values)):
   sel = filter(lambda x: x[-1]==c, points)
   x = [s[0] for s in sel]
   y = [s[1] for s in sel]
   scatter(x, y, c=colors[c])
savefig('mds.png', dpi=72)
