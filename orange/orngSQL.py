#!/usr/bin/python
# A module to read data from an SQL database into an Orange ExampleTable.
# The goal is to keep it compatible with PEP 249.
# For now, the writing shall be basic, if it works at all.

import orange

import urllib

class SQLReader:
    def __init__(addr = None):
        if addr:
            connect(addr)
        self.domainDepot = orange.DomainDepot()
        self.domain = None

    def _parseURI(uri):
        """ lifted straight from sqlobject """
        schema, rest = uri.split(':', 1)
        assert rest.startswith('/'), "URIs must start with scheme:/ -- you did not include a / (in %r)" % rest
        if rest.startswith('/') and not rest.startswith('//'):
            host = None
            rest = rest[1:]
        elif rest.startswith('///'):
            host = None
            rest = rest[3:]
        else:
            rest = rest[2:]
            if rest.find('/') == -1:
                host = rest
                rest = ''
            else:
                host, rest = rest.split('/', 1)
        if host and host.find('@') != -1:
            user, host = host.split('@', 1)
            if user.find(':') != -1:
                user, password = user.split(':', 1)
            else:
                password = None
        else:
            user = password = None
        if host and host.find(':') != -1:
            _host, port = host.split(':')
            try:
                port = int(port)
            except ValueError:
                raise ValueError, "port must be integer, got '%s' instead" % port
            if not (1 <= port <= 65535):
                raise ValueError, "port must be integer in the range 1-65535, got '%d' instead" % port
            host = _host
        else:
            port = None
        path = '/' + rest
        if os.name == 'nt':
            if (len(rest) > 1) and (rest[1] == '|'):
                path = "%s:%s" % (rest[0], rest[2:])
        args = {}
        if path.find('?') != -1:
            path, arglist = path.split('?', 1)
            arglist = arglist.split('&')
            for single in arglist:
                argname, argvalue = single.split('=', 1)
                argvalue = urllib.unquote(argvalue)
                args[argname] = argvalue
        return schema, user, password, host, port, path, args

    def connect(uri):
        """the uri string's syntax is the same as that of sqlobject.
        Unfortunately, only postgres and mysql are going to be supported in
        the near future.
        scheme://[user[:password]@]host[:port]/database[?parameters]
        Examples:
        mysql://user:password@host/database
        mysql://host/database?debug=1
        postgres://user@host/database?debug=&cache=
        postgres:///full/path/to/socket/database
        postgres://host:5432/database
        """
        (schema, user, password, host, port, path, args) = self._parseURI(uri)
        if schema == 'postgres':
            import psycopg2 as dbmod
        elif schema == 'mysql':
            import MySQLdb as dbmod
        self.dbmod = dbmod
        dbArgDict = {}
        if user:
            dbArgDict['user'] = user
        if password:
            dbArgDict['password'] = password
        if host:
            dbArgDict['host'] = host
        if port:
            dbArgDict['port'] = port
        if path:
            dbArgDict['database'] = path
        self.conn = self.dbmod.connect(**dbArgDict)
        
    def query(s, discreteAttrs = None, classAttr = None, metaAttrs = None):
        """executes a read query and constructs the orange.Domain.
        If the name of the class attribute is not supplied, the first column is used as the class attribute.
        If any of the attributes are considered to be discrete, they should be specified.
        """
        try:
            self.curs = self.conn.cursor()
            self.curs.execute(s)
            desc = self.curs.desctiption
            if not classAttr:
                classAttr = desc[0][0]
            attrNames = []
            for i in desc:
                name = i[0]
                typ = i[1]
                if typ in [self.dbmod.STRING, self.dbmod.DATETIME]:
                    attrName = 'S#' + name
                elif name in discreteAttrs:
                    attrName = 'D#' + name
                else:
                    attrName = 'C#' + name
                if name == classAttr:
                    attrName = "c" + attrName
                elif name in metaAttrs:
                    attrName = "m" + attrName
                attrNames.append(attrName)
            self.domain = self.domainDepot.prepareDomain(attrNames)
        except:
            self.domain = None
    def data():
        data = None
        if self.domain:
            data = orange.ExampleTable(self.domain)
            r = self.curs.fetchone()
            while r:
                example = orange.Example(self.domain, [str(i) for i in r])
                data.append(example)
                r = self.curs.fetchone()
        return data
    
class SQLWriter:
    pass