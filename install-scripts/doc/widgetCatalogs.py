def getWidgetsByCategory(xmlfilename):
    categories = {}
    from xml.dom import minidom
    for categoryNode in minidom.parse(open(xmlfilename)).getElementsByTagName("category"):
        category = categoryNode.getAttribute("name")
        if category == "Prototypes" or categoryNode.hasAttribute("directory"):
            continue
        for widgetNode in categoryNode.getElementsByTagName("widget"):
            categories.setdefault(category, []).append(dict([(x, widgetNode.getAttribute(x)) for x in ["name", "contact", "icon", "priority", "file", "in", "out"]]))
    for cw in categories.values():
        cw.sort(lambda x,y: cmp(int(x["priority"]), int(y["priority"])) or cmp(x["name"], y["name"]))
    return categories


def mergeCategories(categoriesOrder, xmlCategories):
    return categoriesOrder + [x for x in xmlCategories.keys() if x not in categoriesOrder]


def createCanvasCatalogPage(xmlCategories, docpath =".", categoriesOrder = ["Data", "Visualize", "Classify", "Evaluate", "Associate", "Regression"], verbose=False):
    from os.path import exists
    
    catalogPage = "<table>"
    if docpath[-1] not in "\\/":
        docpath += "/"
    
    for category in mergeCategories(categoriesOrder, xmlCategories):
        catalogPage += '<tr><td COLSPAN="6" style="border-bottom: 2px solid #F8CB66; padding-left: 4px; font-weight:bold; padding-top: 12px; padding-bottom: 4px; margin-bottom: 12px;">%s</td></tr>\n\n\n' % category
        catalogPage += '<tr valign="top">\n'
        for i, widget in enumerate(xmlCategories[category]):
            if i and not i % 6:
                catalogPage += '</tr><tr valign="top">'
            name = widget["name"]
            namep = name.replace(" ", "")
            htmlfile = docpath + category + "/" + namep + ".htm"
            icon = widget["icon"]
            if not exists(docpath + icon):
                icon = "icons/" + namep + ".png"
                if not exists(docpath + icon):
                    icon = "icons/Unknown.png"
            if verbose:
                contact = widget["contact"]
                if "(" in contact:
                    widget["contact"] = contact[:contact.index("(")]
                verb = '<br/><font color="#bbbbbb"><small><br>%(file)s, %(priority)s<br>%(contact)s<br></small></font>' % widget
            else:
                verb = ""
            if exists(htmlfile):
                catalogPage += '<td align="center" style="padding-bottom: 12px; padding-top: 6px"><a href="%s"><img src="%s"><br/>%s</a>%s</td>' % (htmlfile, icon, name, verb)
            else:
                catalogPage += '<td align="center" style="padding-bottom: 12px; padding-top: 6px"><img src="%s"><br/><FONT COLOR="#bbbbbb">%s</FONT>%s</td>\n\n' % (icon, name, verb)
        catalogPage += '</tr>'

    catalogPage += "</table>"
    
    return catalogPage


def createHHStructure(xmlCategories, docpath = ".", categoriesOrder = ["Data", "Visualize", "Classify", "Evaluate", "Associate", "Regression"]):
    from os.path import exists
    
    hhStructure = "Widget Catalog ---> default.htm\n"
    
    if docpath[-1] not in "\\/":
        docpath += "/"

    for category in mergeCategories(categoriesOrder, xmlCategories):
        introduction = "\t%s ---> default.htm\n" % category
        for widget in xmlCategories[category]:
            name = widget["name"]
            catnamep = category + "/" + name.replace(" ", "") + ".htm"
            if exists(docpath + catnamep):
                hhStructure += introduction
                introduction = ""
                hhStructure += "\t\t%s ---> %s\n" % (name, catnamep)
                
    return hhStructure                

if __name__=="__main__":
    from sys import argv
    categories = getWidgetsByCategory(argv[2])
    docpath = len(argv) > 3 and argv[3] or "."
    
    if argv[1] == "hh":
        print createHHStructure(categories, docpath)
    elif argv[1] == "html":
        print createCanvasCatalogPage(categories, docpath)
    elif argv[1] == "htmlverb":
        print createCanvasCatalogPage(categories, docpath, verbose=True)
