import orange, time

class DateVariable(orange.PythonVariable):
    def str2val(self, str):
        return time.strptime(str, "%b %d %Y")

    def val2str(self, val):
        return time.strftime("%b %d %Y (%a)", val)


birth = DateVariable("birth")
val = birth("Aug 19 2003")
print val

data = orange.ExampleTable("lenses")

newdomain = orange.Domain(data.domain.attributes + [birth], data.domain.classVar)
newdata = orange.ExampleTable(newdomain, data)

newdata[0]["birth"] = "Aug 19 2003"
print newdata[0]

orange.saveTabDelimited("del2", newdata)

print newdata[0]

orange.saveTabDelimited("del2", newdata)

newdata[0]["birth"] = "Aug 19 2003"
newdata[1]["birth"] = "Jan 12 1998"
newdata[2]["birth"] = "Sep 1 1995"
newdata[3]["birth"] = "May 25 2001"
newdata.sort("birth")
print "\nSorted data"
for i in newdata:
    print i

    