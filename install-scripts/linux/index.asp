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

<pre>
<%=includeOrNotFound("windows.regress.log")%>
</pre>

</BODY>
</HTML>
