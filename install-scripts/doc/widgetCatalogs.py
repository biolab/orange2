def getWidgetsByCategory(xmlfilename):
    categories = {}
    from xml.dom import minidom
    for categoryNode in minidom.parse(open(xmlfilename)).getElementsByTagName("category"):
        category = categoryNode.getAttribute("name")
        for widgetNode in categoryNode.getElementsByTagName("widget"):
            categories.setdefault(category, []).append(dict([(x, widgetNode.getAttribute(x)) for x in ["name", "contact", "icon", "priority"]]))
    for cw in categories.values():
        cw.sort(lambda x,y: cmp(x["priority"], y["priority"]) or cmp(x["name"], y["name"]))
    return categories


def mergeCategories(categoriesOrder, xmlCategories):
    return categoriesOrder + filter(lambda x:x not in categoriesOrder, xmlCategories.keys())


def createCanvasCatalogPage(xmlCategories, docpath =".", categoriesOrder = ["Data", "Visualize", "Classify", "Evaluate", "Associate", "Regression"]):
    from os.path import exists
    
    catalogPage = "<table>"
    if docpath[-1] not in "\\/":
        docpath += "/"
    
    for category in mergeCategories(categoriesOrder, xmlCategories):
        catalogPage += '<tr><td COLSPAN="6"><H2>%s</H2></td></tr>\n\n\n' % category
        catalogPage += '<tr>\n'
        for i, widget in enumerate(xmlCategories[category]):
            if i and not i % 3:
                catalogPage += '</tr><tr>'
            htmlfile = docpath + category + "/" + widget["name"].replace(" ", "") + ".htm"
            if exists(htmlfile):
                catalogPage += '<td><a href="%s"><img src="%s"></a></td>\n' % (htmlfile, widget["icon"]) + \
                               '<td style="padding-right: 15"><a href="%s">%s</a></td>\n\n' % (htmlfile, widget["name"])
            else:
                catalogPage += '<td><img style="padding: 2;" src="%s"></td>\n' % widget["icon"] + \
                               '<td style="padding-right: 15"><FONT COLOR="#bbbbbb">%s</FONT></a></td>\n\n' % widget["name"]
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
    else:
        print createCanvasCatalogPage(categories, docpath)
