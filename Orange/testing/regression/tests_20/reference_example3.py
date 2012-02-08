# Description: Shows some more uses for meta-attributes with orange.Example
# Category:    basic classes, meta-attributes
# Classes:     Example
# Uses:        lenses
# Referenced:  Example.htm

import orange, random

data = orange.ExampleTable("lenses")
random.seed(0)
#id2 = orange.newmetaid()
#w2 = orange.FloatVariable("ww")
#The below two lines fail (and SHOULD fail):
#data[0].setmeta(id, orange.Value(ww, 2.0))
#data[0].setmeta(id2, "2.0")

ok_id = orange.newmetaid()
ok = orange.EnumVariable("ok?", values=["no", "yes"])

data[0][ok_id] = orange.Value(ok, "yes")

data.domain.addmeta(ok_id, ok)

data[0][ok_id] = "yes"
data[0][ok] = "no"
data[0]["ok?"] = "no"

no_yes = [orange.Value(ok, "no"), orange.Value(ok, "yes")]
for example in data:
    example.setvalue(no_yes[random.randint(0, 1)])

print data[0][ok_id]
print data[0][ok]
print data[0]["ok?"]

data[0].removemeta(ok_id)
data[1].removemeta(ok)
data[2].removemeta("ok?")

w = orange.FloatVariable("w")
w_id = orange.newmetaid()
data.domain.addmeta(w_id, w)
data[0].setweight(w, 1)
data[1].setweight("w", 2)
data[2].setweight(w_id, 3)
data[3].setweight(0, 4)
data[4].setweight(None, 5)

print "Some weights..."
for example in data[:6]:
    print example

data[0].removeweight(w)
data[1].removeweight("w")
data[2].removeweight(w_id)
data[3].removeweight(0)
data[4].removeweight(None)

print "\n\n... and without them"
for example in data[:6]:
    print example

