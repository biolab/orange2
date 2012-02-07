import orange

import time

class DateValue(orange.SomeValue):
    def __init__(self, date):
        self.date = date
        
    def __str__(self):
        return time.strftime("%b %d %Y (%a)", self.date)

    def __cmp__(self, other):
        return cmp(self.date, other.date)
    
class DateVariable(orange.PythonVariable):
    def str2val(self, str):
        return DateValue(time.strptime(str, "%b %d %Y"))

birth = DateVariable("birth")
val = birth("Aug 19 2003")
print val

data = orange.ExampleTable("lenses")

newdomain = orange.Domain(data.domain.attributes + [birth], data.domain.classVar)
newdata = orange.ExampleTable(newdomain, data)

newdata[0]["birth"] = "Aug 19 2003"
newdata[1]["birth"] = "Jan 12 1998"
newdata[2]["birth"] = "Sep 1 1995"
newdata[3]["birth"] = "May 25 2001"
newdata.sort("birth")
print "\nSorted data"
for i in newdata:
    print i
