<%@ Language=Python %>
<%
import os, time

def timeLinkOrNotFound(fname, desc):
	return os.path.exists(fname) and "<a href=%s>%s</a> (%s)" % (fname, desc, time.ctime(os.path.getmtime(fname))) or "%s (not found)" % (desc)

def includeOrNotFound(fname):
	return os.path.exists(fname) and file(fname, "rt").read() or "not found!"
%>

<HTML>
<HEAD>
<TITLE>Orange daily build report</TITLE>
<link rel="shortcut icon" href="orange.ico">
</HEAD>
<BODY>

<table>
<tr>
<td valign=top><b>Linux</b></td>
<td>
 <P><%=timeLinkOrNotFound("linux.output.log", "Full log for Orange daily build")%></p>
 <p><%=timeLinkOrNotFound("linux.compiling.log", "compile log")%><br>
 <%=timeLinkOrNotFound("linux.install.log", "local install log")%><br>
 <%=timeLinkOrNotFound("linux.regress.log", "regression tests log")%></p>
</td>
</tr>
<tr><td>&nbsp;</td></tr>
<tr><td valign=top><b>Windows&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b></td>
<td>
 <p><a href="index.asp#windowsLog">Full log for Orange daily build</a></p>
 <p><%=timeLinkOrNotFound("windows.compiling.log", "compile log (daily snapshots)")%><br>
 <table><tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td><td>
<%
for ver in range(23, 26):
    fn = "win-compile-%s.log" % ver
    if os.path.exists(fn):
        l1 = None
        for l in file(fn):
            l2 = l1
            l1 = l
        Response.Write('<a href="index.asp#win-compile-py%i">Build for Python %2.1f</a> (%s; %s)<br/>\n' % (ver, ver/10., time.ctime(os.path.getmtime(fn)), l2.replace("=","").replace("Build: ", "").strip()))
    else:
        Response.Write('Build for Python %2.1f: log file not found<br/>\n' % (ver/10., ))
%>
  </td></tr></table>
<%=timeLinkOrNotFound("windows.regress.log", "install and regression tests log")%></P>
</td></tr>
</table>

<P>&nbsp;</P>
<P><a href="regressionTests">Regression test output files</a></p>
<P>&nbsp;</P>

<h3>Full Log on Linux</h3>
<pre>
<%=includeOrNotFound("linux.output.log")%>
</pre>

<a name="windowsLog"></a>
<h3>Full Log on Windows</h3>
<pre>
<%=includeOrNotFound("windows.compiling.log")%>
</pre>

<%
for ver in range(23, 26):
    fname = "win-compile-%s.log" % ver
    if os.path.exists(fn):
        Response.Write('<a name="win-compile-py%i"></a><h3>Compiling for Python %2.1f</h3><pre>%s</pre>' % (ver, ver/10., file(fname, "rt").read()))
        

%>
<h3>Regression tests</h3>

<pre>
<%=includeOrNotFound("windows.regress.log")%>
</pre>

</BODY>
</HTML>
