# A module to read data from an SQL database into an Orange ExampleTable.
# The goal is to keep it compatible with PEP 249.
# For now, the writing shall be basic, if it works at all.

import orange
import os
import urllib

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

class __DummyQuirkFix:
    def __init__(self, dbmod):
        self.dbmod = dbmod
        self.typeDict = {
            orange.VarTypes.Continuous:'FLOAT', 
            orange.VarTypes.Discrete:'VARCHAR(250)', orange.VarTypes.String:'VARCHAR(250)'}
    def beforeWrite(self, cursor):
        pass
    def beforeCreate(self, cursor):
        pass
    def beforeRead(self, cursor):
        pass
class __MySQLQuirkFix(__DummyQuirkFix):
    def __init__(self, dbmod):
        self.dbmod = dbmod
        self.BOOLEAN = None
        self.STRING = dbmod.STRING
        self.DATETIME = dbmod.DATETIME
        self.typeDict = {
            orange.VarTypes.Continuous:'DOUBLE', 
            orange.VarTypes.Discrete:'VARCHAR(250)', orange.VarTypes.String:'VARCHAR(250)'}
    def beforeWrite(self, cursor):
        cursor.execute("SET sql_mode='ANSI_QUOTES';")
    def beforeCreate(self, cursor):
        cursor.execute("SET sql_mode='ANSI_QUOTES';")
    def beforeRead(self, cursor):
        pass
class __PostgresQuirkFix(__DummyQuirkFix):
    def __init__(self, dbmod):
        self.dbmod = dbmod
        self.BOOLEAN = 16
        self.STRING = dbmod.STRING
        self.DATETIME = dbmod.DATETIME
        self.typeDict = {
            orange.VarTypes.Continuous:'FLOAT', 
            orange.VarTypes.Discrete:'VARCHAR', orange.VarTypes.String:'VARCHAR'}
    def beforeWrite(self, cursor):
        pass
    def beforeCreate(self, cursor):
        pass
    def beforeRead(self, cursor):
        pass

def _connection(uri):
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
        (schema, user, password, host, port, path, args) = _parseURI(uri)
        if schema == 'postgres':
            import psycopg2 as dbmod
            argTrans = {
            'host':'host',
            'port':'port',
            'user':'user',
            'password':'password',
            'database':'database'
            }
            quirks = __PostgresQuirkFix(dbmod)
        elif schema == 'mysql':
            import MySQLdb as dbmod
            argTrans = {
            'host':'host',
            'port':'port',
            'user':'user',
            'password':'passwd',
            'database':'db'
            }
            quirks = __MySQLQuirkFix(dbmod)
        dbArgDict = {}
        if user:
            dbArgDict[argTrans['user']] = user
        if password:
            dbArgDict[argTrans['password']] = password
        if host:
            dbArgDict[argTrans['host']] = host
        if port:
            dbArgDict[argTrans['port']] = port
        if path:
            dbArgDict[argTrans['database']] = path[1:]
        return (quirks, dbmod.connect(**dbArgDict))

class SQLReader(object):
    def __init__(self, addr = None, domainDepot = None):
        if addr is not None:
            self.connect(addr)
        if domainDepot is not None:
            self.domainDepot = domainDepot
        else:
            self.domainDepot = orange.DomainDepot()
        self.exampleTable = None
        self._dirty = True
    def connect(self, uri):
        self._dirty = True
        self.delDomain()
        (self.quirks, self.conn) = _connection(uri)
    def disconnect(self):
        self.conn.disconnect()
    def getClassName(self):
        self.update()
        return self.domain.classVar.name
    def setClassName(self, className):
        self._className = className
        self.delDomain()
    def delClassName(self):
        del self._className
    className = property(getClassName, setClassName, delClassName, "the name of the class variable")

    def getMetaNames(self):
        self.update()
        return self.domain.getmetas().values()
    def setMetaNames(self, metaNames):
        self._metaNames = metaNames
        self.delDomain()
    def delMetaNames(self):
        del self._metaNames
    metaNames = property(getMetaNames, setMetaNames, delMetaNames, "the names of the meta attributes")

    def setDiscreteNames(self, discreteNames):
        self._discreteNames = discreteNames
        self.delDomain()
    def getDiscreteNames(self):
        self.update()
        return self._discreteNames
    def delDiscreteNames(self):
        del self._discreteNames
    discreteNames = property(getDiscreteNames, setDiscreteNames, delDiscreteNames, "the names of the discrete attributes")

    def setQuery(self, query, domain = None):
        """sets the query, resets the internal variables, without executing the query"""
        self._query = query
        self._dirty = True
        if domain is not None:
            self._domain = domain
        else:
            self.delDomain()
    def getQuery(self):
        return self._query
    def delQuery(self):
        del self._query
    query = property(getQuery, setQuery, delQuery, "The query to be executed on the next execute()")
    def generateDomain(self):
        pass
    def setDomain(self, domain):
        self._domain = domain
        self._dirty = True
    def getDomain(self):
        if not hasattr(self, '_domain'):
            self._createDomain()
        return self._domain
    def delDomain(self):
        if hasattr(self, '_domain'):
            del self._domain
    domain = property(getDomain, setDomain, delDomain, "the Orange domain")
    def execute(self, query, domain = None):
        """executes an sql query"""
        self.setQuery(query, domain)
        self.update()
        
    def _createDomain(self):
        if hasattr(self, '_domain'):
            return
        attrNames = []
        if not hasattr(self, '_discreteNames'):
            self._discreteNames = []
        discreteNames = self._discreteNames
        if not hasattr(self, '_metaNames'):
            self._metaNames = []
        metaNames = self._metaNames
        if not hasattr(self, '_className'):
            className = None
        else:
            className = self._className
        for i in self.desc:
            name = i[0]
            typ = i[1]
            if name in discreteNames or typ == self.quirks.BOOLEAN:
                attrName = 'D#' + name
            elif typ == self.quirks.STRING:
                    attrName = 'S#' + name
            elif typ == self.quirks.DATETIME:
                attrName = 'S#' + name
            else:
                attrName = 'C#' + name
            if name == className:
                attrName = "c" + attrName
            elif name in metaNames:
                attrName = "m" + attrName
            elif not className and name == 'class':
                attrName = "c" + attrName
            attrNames.append(attrName)
    #       print "NAME:", '"%s"' % name, ", t:", typ, " attrN:", '"%s"' % attrName
        (self._domain, self._metaIDs, dummy) = self.domainDepot.prepareDomain(attrNames)
 #           print "Created domain."
        del dummy

        
    def update(self):
        if not self._dirty and hasattr(self, '_domain'):
            return self.exampleTable
        self.exampleTable = None
        try:
            curs = self.conn.cursor()
            try:
                self.quirks.beforeRead(curs)
                curs.execute(self.query)
            except Exception, e:
                self.conn.rollback()
                raise e
            self.desc = curs.description
            # for reasons unknown, the attributes get reordered.
            domainIndexes = [0] * len(self.desc)
            self._createDomain()
            attrNames = []
            for i, name in enumerate(self.desc):
            #    print name[0], '->', self.domain.index(name[0])
                domainIndexes[self._domain.index(name[0])] = i
                attrNames.append(name[0])
            self.exampleTable = orange.ExampleTable(self.domain)
            r = curs.fetchone()
            while r:
                # for reasons unknown, domain rearranges the properties
                example = orange.Example(self.domain)
                for i in xrange(len(r)):
                    val = str(r[i])
                    var = example[attrNames[i]].variable
                    if type(var) == orange.EnumVariable and val not in var.values:
                        var.values.append(val)
                    example[attrNames[i]] = str(r[i])
                self.exampleTable.append(example)
                r = curs.fetchone()
            self._dirty = False
        except Exception, e:
            self.domain = None
            raise
            #self.domain = None

    def data(self):
        self.update()
        if self.exampleTable:
            return self.exampleTable
        return None
    
class SQLWriter(object):
    def __init__(self, uri = None):
        if uri is not None:
            self.connect(uri)
    
    def connect(self, uri):
        (self.quirks, self.connection) = _connection(uri)
    def __attrVal2sql(self, d):
        if d.varType == orange.VarTypes.Continuous:
            return d.value
        elif d.varType == orange.VarTypes.Discrete:
            return str(d.value)
        else:
            return "'%s'" % str(d.value)
    def __attrName2sql(self, d):
        return d.name
    def __attrType2sql(self, d):
        return self.quirks.typeDict[d]
    def write(self, table, data, renameDict = None):
        """if provided, renameDict maps the names in data to columns in
        the database. For each var in data: dbColName = renameDict[var.name]"""
        l = [i.name for i in data.domain.attributes]
        l += [i.name for i in data.domain.getmetas().values()]
        if data.domain.classVar:
            l.append(data.domain.classVar.name)
        if renameDict is None:
            renameDict = {}
        colList = []
        for i in l:
            colList.append(renameDict.get(str(i), str(i)))
        try:
            cursor=self.connection.cursor()
            self.quirks.beforeWrite(cursor)
            query = 'INSERT INTO "%s" (%s) VALUES (%s);'
            for d in data:
                valList = []
                colSList = []
                for (i, name) in enumerate(colList):
                    colSList.append('"%s"'% name)
                    valList.append(self.__attrVal2sql(d[l[i]]))
                valStr = ', '.join(["%s"]*len(colList))
                # print "exec:", query % (table, "%s ", "%s "), tuple(colList + valList)
                cursor.execute(query % (table, 
                    ", ".join(colSList), 
                    ", ".join (["%s"] * len(valList))), tuple(valList))
            cursor.close()
            self.connection.commit()
        except Exception, e:
            import traceback
	    traceback.print_exc()
            self.connection.rollback()

    def create(self, table, data, renameDict = None, typeDict = None):
        l = [(i.name, i.varType ) for i in data.domain.attributes]
        l += [(i.name, i.varType ) for i in data.domain.getmetas().values()]
        if data.domain.classVar:
            l.append((data.domain.classVar.name, data.domain.classVar.varType))
        if renameDict is None:
            renameDict = {}
        colNameList = [renameDict.get(str(i[0]), str(i[0])) for i in l]
        if typeDict is None:
            typeDict = {}
        colTypeList = [typeDict.get(str(i[0]), self.__attrType2sql(i[1])) for i in l]
        try:
            cursor = self.connection.cursor()
            colSList = []
            for (i, name) in enumerate(colNameList):
                colSList.append('"%s" %s' % (name, colTypeList[i]))
            colStr = ", ".join(colSList)
            query = """CREATE TABLE "%s" ( %s );""" % (table, colStr)
            self.quirks.beforeCreate(cursor)
            cursor.execute(query)
            print query
            self.write(table, data, renameDict)
            self.connection.commit()
        except Exception, e:
            self.connection.rollback()
    
    def disconnect(self):
        self.conn.disconnect()

def loadSQL(filename, dontCheckStored = False, domain = None):
    f = open(filename)
    lines = f.readlines()
    queryLines = []
    discreteNames = None
    uri = None
    metaNames = None
    className = None
    for i in lines:
        if i.startswith("--orng"):
            (dummy, command, line) = i.split(None, 2)
            if command == 'uri':
                uri = eval(line)
            elif command == 'discrete':
                discreteNames = eval(line)
            elif command == 'meta':
                metaNames = eval(line)
            elif command == 'class':
                className = eval(line)
            else:
                queryLines.append(i)
        else:
            queryLines.append(i)
    query = "\n".join(queryLines)
    r = SQLReader(uri)
    if discreteNames:
        r.discreteNames = discreteNames
    if className:
        r.className = className
    if metaNames:
        r.metaNames = metaNames
    r.execute(query)
    data = r.data()
    return data

def saveSQL():
    pass