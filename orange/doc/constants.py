hhphead = """
[OPTIONS]
Compiled file=orange.chm
Contents file=orange.hhc
Default topic=./default.htm
Display compile progress=No
Full text search stop list file=../stop.stp
Full-text search=Yes
Index file=orange.hhk
Language=0x409
Title=Orange Documentation
"""

hhchead = """
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
<HTML>
<HEAD>
<meta name="GENERATOR" content="Microsoft&reg; HTML Help Workshop 4.1">
<!-- Sitemap 1.0 -->
</HEAD><BODY>
<OBJECT type="text/site properties">
	<param name="Window Styles" value="0x801227">
	<param name="ImageType" value="Folder">
</OBJECT>
<UL>
"""

hhcentry = """%(spc)s<LI><OBJECT type="text/sitemap">
%(spc)s    <param name="Name" value="%(name)s">
%(spc)s    <param name="Local" value="%(file)s">
%(spc)s</OBJECT>
"""


hhkhead = """
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
<HTML>
<HEAD>
<meta name="GENERATOR" content="Microsoft&reg; HTML Help Workshop 4.1">
<!-- Sitemap 1.0 -->
</HEAD><BODY>
<UL>
"""

hhkentry = """<LI><OBJECT type="text/sitemap">
    <param name="Name" value="%s">
"""

hhksubentry = """
    <param name="Name" value="%s">
    <param name="Local" value="%s#HH%i">
"""

hhkendentry = "</OBJECT>\n\n"


jh_idx = """
<?xml version='1.0' encoding='ISO-8859-1' ?>
<!DOCTYPE index
  PUBLIC "-//Sun Microsystems Inc.//DTD JavaHelp Index Version 1.0//EN" "index_1_0.dtd">

<index version="1.0">
"""

jh_toc = """
<?xml version='1.0' encoding='ISO-8859-1' ?>
<!DOCTYPE toc
  PUBLIC "-//Sun Microsystems Inc.//DTD JavaHelp Index Version 1.0//EN" "toc_2_0.dtd">

<toc version="1.0">
"""

jh_map = """
<?xml version='1.0' encoding='ISO-8859-1' ?>
<!DOCTYPE map
  PUBLIC "-//Sun Microsystems Inc.//DTD JavaHelp Map Version 1.0//EN"
  "http://java.sun.com/products/javahelp/map_1_0.dtd">
  
<map version="1.0">
"""