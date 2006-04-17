<%@ LANGUAGE = PYTHON %>

<%
baspath = Server.MapPath("/orange/doc/widgets/catalog")

def insertFile(cat, name):
    import os
    namep = name.replace(" ", "")
    s = cat + "/" + name
    if os.path.exists(baspath+"\\"+s+".htm"):
        Response.Write('<tr><td><a href="%s.htm"><img src="icons/%s.png"></a></td>\n' % (s, namep) + \
                       '<td><a href="%s.htm">%s</a></td></tr>\n\n' % (s, name))
    else:
        Response.Write('<tr><td><img style="padding: 2;" src="icons/%s.png"></td>\n' % namep + \
                       '<td><FONT COLOR="#bbbbbb">%s</FONT></a></td></tr>\n\n' % name)

def category(cat, names):
    Response.Write('<H2>Data</H2>\n<table>\n\n')
    for name in names:
        insertFile(cat, name)
    Response.Write('</table>')
    
def emptyColumn():
    Response.Write('</td><td>&nbsp;&nbsp;&nbsp;</td><td valign="top">')
%>

<body>
<h1>Catalog of Orange Widgets</h1>

<p>Orange Widgets are building blocks of Orange's graphical user's
interface and its visual programming interface. The purpose of this
documention is to describe the basic functionality of the widgets,
and show how are they used in interaction with other widgets.</p>

<p>Widgets in Orange can be used on their own, or within a separate
application. This documention will however describe them as used
within Orange Canvas, and application which trough visual programming
allows gluing of widgets together in whatcan be anything from simple
data analysis schema to a complex explorative data analysis
application.</p>

<p>In Orange Canvas, widgets are grouped according to their
functionality. We stick to the same grouping in this documentation,
and cluster widgets accoring to their arrangement withing Canvas's
toolbar.</p>

<P>The documentation refers to the last snapshot of Orange. The
version you use might miss some stuff which is already described
here. Download the new snapshot if you need it.</P>

<table>
<tr><td valign="top">

<%
category("Data", ["File", "Save", "Data Table", "Select Attributes", "Data Sampler", "Select Data", "Discretize", "Rank"])

emptyColumn()

category("Visualize", ["Attribute Statistics", "Distributions", "Scatterplot", "Scatterplot Matrix",
            "Radviz", "Polyviz", "Parallel Coordinates", "Survey Plot",
            "Sieve Diagram", "Mosaic Display", "Sieve Multigram"])

emptyColumn()

category("Associate", ["Association Rules", "Association Rules Filter", "Association Rules Viewer", "Association Rules Print",
          "Example Distance", "Attribute Distance", "Distance Map", "K-means Clustering",
          "Interaction Graph", "MDS", "Hierarchical Clustering"])
%>

</tr>
</table>

<br><br>

<table>
<tr><td valign="top">

<%
category("Classify", ["Naive Bayes", "Logistic Regression", "Majority", "k-Nearest Neighbours", "Classification Tree",
                         "C4.5", "Interactive Tree Builder", "SVM", "CN2", "Classification Tree Viewer",
                         "Classification Tree Graph", "CN2 Rules Viewer", "Nomogram"])

emptyColumn()

category("Evaluate", ["Test Learners", "Classifications", "ROC Analysis", "Lift Curve", "Calibration Plot"])
%>

</tr>
</table>

</body></html>
