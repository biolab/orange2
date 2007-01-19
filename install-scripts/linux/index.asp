<%@ Language=Python %>
<%
import os, time

def timeOrNotFound(fname):
	return os.path.exists(fname) and time.ctime(os.path.getmtime(fname)) or "not found!"

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
 <P><a href="linux.output.log">Full log for Orange daily build</a> (<%=timeOrNotFound("linux.output.log")%>)</p>
 <p><a href="linux.compiling.log">compile log</a> (<%=timeOrNotFound("linux.compiling.log")%>)<br>
 <a href="linux.install.log">local install log</a> (<%=timeOrNotFound("linux.install.log")%>)<br>
 <a href="linux.regress.log">regression tests log</a> (<%=timeOrNotFound("linux.regress.log")%>)
</td>
</tr>
<tr><td>&nbsp;</td></tr>
<tr><td valign=top><b>Windows&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b></td>
<td>
 <p><a href="windows.output.log">Full log for Orange daily build</a> (<%=timeOrNotFound("windows.output.log")%>)</p>
 <p><a href="windows.compiling.log">compile log</a> (<%=timeOrNotFound("windows.compiling.log")%>)<br>
 <a href="windows.install.log">local install log</a> (<%=timeOrNotFound("windows.install.log")%>)<br>
 <a href="windows.regress.log">regression tests log</a> (<%=timeOrNotFound("windows.regress.log")%>)</P>
<P>
<a href="index.asp#windowsLog">(Jump to Windows log file)</a></P>
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
<%=includeOrNotFound("windows.output.log")%>
</pre>

</BODY>
</HTML>
