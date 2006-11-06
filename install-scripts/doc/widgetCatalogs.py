def getWidgetsByCategory(xmlfilename):
    categories = {}
    from xml.dom import minidom
    for categoryNode in minidom.parse(open(xmlfilename)).getElementsByTagName("category"):
        category = categoryNode.getAttribute("name")
        for widgetNode in categoryNode.getElementsByTagName("widget"):
            categories.setdefault(category, []).append(dict([(x, widgetNode.getAttribute(x)) for x in ["name", "contact", "icon", "priority", "file", "in", "out"]]))
    for cw in categories.values():
        cw.sort(lambda x,y: cmp(int(x["priority"]), int(y["priority"])) or cmp(x["name"], y["name"]))
    return categories


def mergeCategories(categoriesOrder, xmlCategories):
    dontAdd = categoriesOrder + ["Genomics", "Other"]
    return categoriesOrder + filter(lambda x:x not in dontAdd, xmlCategories.keys())


def createCanvasCatalogPage(xmlCategories, docpath =".", categoriesOrder = ["Data", "Visualize", "Classify", "Evaluate", "Associate", "Regression"], verbose=False):
    from os.path import exists
    
    catalogPage = "<table>"
    if docpath[-1] not in "\\/":
        docpath += "/"
    
    for category in mergeCategories(categoriesOrder, xmlCategories):
        catalogPage += '<tr><td style="padding-top:32px" COLSPAN="6"><H2>%s</H2></td></tr>\n\n\n' % category
        catalogPage += '<tr>\n'
        for i, widget in enumerate(xmlCategories[category]):
            if i and not i % 3:
                catalogPage += '</tr><tr>'
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
                verb = '<font color="#bbbbbb"><small><br>%(file)s, %(priority)s<br>%(contact)s<br></small></font>' % widget
            if exists(htmlfile):
                catalogPage += '<td><a href="%s"><img src="%s"></a></td>\n' % (htmlfile, icon) + \
                               '<td style="padding-right: 15"><a href="%s">%s</a>%s</td>\n\n' % (htmlfile, name, verb)
            else:
                catalogPage += '<td><img style="padding: 2;" src="%s"></td>\n' % icon + \
                               '<td style="padding-right: 15"><FONT COLOR="#bbbbbb">%s</FONT></a>%s</td>\n\n' % (name, verb)
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
