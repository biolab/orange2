from xml.dom import minidom
import re

re_parenth = re.compile(r"\(.*\)")

widgets = {}
people = {}

dom1 = minidom.parse(open("../../OrangeCanvas/widgetregistry.xml", "rt"))
for categoryNode in dom1.getElementsByTagName("category"):
    category = categoryNode.getAttribute("name")
    for widgetNode in categoryNode.getElementsByTagName("widget"):
        widget = widgetNode.getAttribute("name")
        contact = re_parenth.sub("", widgetNode.getAttribute("contact")).strip()
        people.setdefault(contact, {}).setdefault(category, []).append(widget)
        widgets.setdefault(category, []).append((widget, contact))


def sorted(d):
    d.sort()
    return d

print "*** WIDGETS by AUTHORS\n"

for contact in sorted(people.keys()):
    print contact or "<no contact>"
    work = people[contact]
    for category in sorted(work.keys()):
        print "    %s: %s" % (category, ", ".join(sorted(work[category])))
    print

    
print "\n\n\n*** AUTHORS by WIDGETS\n"

for category in sorted(widgets.keys()):
    print category
    categoryWidgets = widgets[category]
    for widget in sorted(categoryWidgets):
        print "    %s: %s" % widget
    print
