# Description: Shows how to specify the symbols for undefined values in tab-delimited files
# Category:    data input
# Classes:     ExampleTable
# Uses:        undefineds
# Referenced:  tabdelimited.htm

import orange
data = orange.ExampleTable("undefineds", DK="GDK", DC="GDC")

for ex in data:
    print ex

print "Default saving\n"
orange.saveTabDelimited("undefined-saved.tab", data)
print open("undefined-saved.tab", "rt").read()

print "Saving with all undefined as NA\n"
orange.saveTabDelimited("undefined-saved-na.tab", data, NA="NA")
print open("undefined-saved.tab", "rt").read()

print "Saving with all undefined as NA\n"
orange.saveTabDelimited("undefined-saved-dc-dk", data, DC="GDC", DK="GDK")
print open("undefined-saved.tab", "rt").read()

import os
os.remove("undefined-saved.tab")