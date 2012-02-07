import orange, time

class DateVariable(orange.PythonVariable):
    def str2val(self, str):
        return time.strptime(str, "%b %d %Y")

    def val2str(self, val):
        return time.strftime("%b %d %Y (%a)", val)

    def filestr2val(self, str, example):
        if str == "unknown":
            return orange.PythonValueSpecial(orange.ValueTypes.DK)
        return DateValue(time.strptime(str, "%m/%d/%Y"))

    def val2filestr(self, val, example):
        return time.strftime("%m/%d/%Y", val)

birth = DateVariable("birth")
val = birth("Aug 19 2003")
print val

data = orange.ExampleTable("lenses")

newdomain = orange.Domain(data.domain.attributes + [birth], data.domain.classVar)
newdata = orange.ExampleTable(newdomain, data)

newdata[0]["birth"] = "Aug 19 2003"
print newdata[0]

orange.saveTabDelimited("del2", newdata)
