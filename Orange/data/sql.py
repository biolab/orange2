import os
import urllib
import Orange
from Orange.misc import deprecated_keywords, deprecated_members
from Orange.feature import Descriptor

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

class __MySQLQuirkFix(object):
    def __init__(self, dbmod):
        self.dbmod = dbmod
        self.typeDict = {
            Descriptor.Continuous:'DOUBLE',
            Descriptor.Discrete:'VARCHAR(250)', Descriptor.String:'VARCHAR(250)'}

    def beforeWrite(self, cursor):
        cursor.execute("SET sql_mode='ANSI_QUOTES';")

    def beforeCreate(self, cursor):
        cursor.execute("SET sql_mode='ANSI_QUOTES';")

class __PostgresQuirkFix(object):
    def __init__(self, dbmod):
        self.dbmod = dbmod
        self.typeDict = {
            Descriptor.Continuous:'FLOAT',
            Descriptor.Discrete:'VARCHAR', Descriptor.String:'VARCHAR'}

    def beforeWrite(self, cursor):
        pass

    def beforeCreate(self, cursor):
        pass

def _connection(uri):
        (schema, user, password, host, port, path, args) = _parseURI(uri)
        argTrans = {
            'host':'host',
            'port':'port',
            'user':'user',
            'password':'passwd',
            'database':'db'
            }
        if schema == 'postgres':
            import psycopg2 as dbmod
            argTrans["database"] = "db"
            quirks = __PostgresQuirkFix(dbmod)
        elif schema == 'mysql':
            import MySQLdb as dbmod
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
    """
    :obj:`~SQLReader` establishes a connection with a database and provides the methods needed
    to fetch the data from the database into Orange.
    """
    @deprecated_keywords({"domainDepot":"domain_depot"})
    def __init__(self, addr = None, domain_depot = None):
        """
        :param uri: Connection string (scheme://[user[:password]@]host[:port]/database[?parameters]).
        :type uri: str

        :param domain_depot: Domain depot
        :type domain_depot: :class:`orange.DomainDepot`
        """
        if uri is not None:
            self.connect(uri)
        if domain_depot is not None:
            self.domainDepot = domain_depot
        else:
            self.domainDepot = orange.DomainDepot()
        self.exampleTable = None
        self._dirty = True

    def connect(self, uri):
        """
        Connect to the database.

        :param uri: Connection string (scheme://[user[:password]@]host[:port]/database[?parameters])
        :type uri: str
        """
        self._dirty = True
        self.delDomain()
        (self.quirks, self.conn) = _connection(uri)

    def disconnect(self):
        """
        Disconnect from the database.
        """
        self.conn.disconnect()

    def getClassName(self):
        self.update()
        return self.domain.class_var.name

    def setClassName(self, className):
        self._className = className
        self.delDomain()

    def delClassName(self):
        del self._className

    class_name = property(getClassName, setClassName, delClassName, "Name of class variable.")
    className = class_name
    
    def getMetaNames(self):
        self.update()
        return self.domain.get_metas().values()

    def setMetaNames(self, meta_names):
        self._metaNames = meta_names
        self.delDomain()

    def delMetaNames(self):
        del self._metaNames

    meta_names = property(getMetaNames, setMetaNames, delMetaNames, "Names of meta attributes.")
    metaName = meta_names

    def setDiscreteNames(self, discrete_names):
        self._discreteNames = discrete_names
        self.delDomain()

    def getDiscreteNames(self):
        self.update()
        return self._discreteNames

    def delDiscreteNames(self):
        del self._discreteNames

    discrete_names = property(getDiscreteNames, setDiscreteNames, delDiscreteNames, "Names of discrete attributes.")
    discreteNames = discrete_names

    def setQuery(self, query, domain = None):
        #sets the query, resets the internal variables, without executing the query
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

    query = property(getQuery, setQuery, delQuery, "Query to be executed on the next execute().")

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

    domain = property(getDomain, setDomain, delDomain, "Orange domain.")

    def execute(self, query, domain = None):
        """
        Executes an sql query.
        """
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
            if name in discreteNames:
                attrName = 'D#' + name
            elif typ == self.quirks.dbmod.STRING:
                    attrName = 'S#' + name
            elif typ == self.quirks.dbmod.DATETIME:
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
        (self._domain, self._metaIDs, dummy) = self.domainDepot.prepareDomain(attrNames)
        del dummy

    def update(self):
        """
        Execute a pending SQL query.
        """
        if not self._dirty and hasattr(self, '_domain'):
            return self.exampleTable
        self.exampleTable = None
        try:
            curs = self.conn.cursor()
            try:
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
            self.exampleTable = Orange.data.Table(self.domain)
            r = curs.fetchone()
            while r:
                # for reasons unknown, domain rearranges the properties
                example = Orange.data.Instance(self.domain)
                for i in xrange(len(r)):
                    val = str(r[i])
                    var = example[attrNames[i]].variable
                    if type(var) == Descriptor.Discrete and val not in var.values:
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
        """
        Return :class:`Orange.data.Table` produced by the last executed query.
        """
        self.update()
        if self.exampleTable:
            return self.exampleTable
        return None

class SQLWriter(object):
    """
    Establishes a connection with a database and provides the methods needed to create
    an appropriate table in the database and/or write the data from an :class:`Orange.data.Table`
    into the database.
    """
    def __init__(self, uri = None):
        """
        :param uri: Connection string (scheme://[user[:password]@]host[:port]/database[?parameters])
        :type uri: str
        """
        if uri is not None:
            self.connect(uri)

    def connect(self, uri):
        """
        Connect to the database.

        :param uri: Connection string (scheme://[user[:password]@]host[:port]/database[?parameters])
        :type uri: str
        """
        (self.quirks, self.connection) = _connection(uri)

    def __attrVal2sql(self, d):
        if d.var_type == Descriptor.Continuous:
            return d.value
        elif d.var_type == Descriptor.Discrete:
            return str(d.value)
        else:
            return "'%s'" % str(d.value)

    def __attrName2sql(self, d):
        return d.name

    def __attrType2sql(self, d):
        return self.quirks.typeDict[d]

    @deprecated_keywords({"renameDict":"rename_dict"})
    def write(self, table, instances, rename_dict = None):
        """
        Writes the data into the table.


        :param table: Table name.
        :type table: str

        :param instances: Data to be written into the database.
        :type instances: :class:`Orange.data.Table`

        :param rename_dict: When ``rename_dict`` is provided the used names are remapped.
            The orange attribute "X" is written into the database column rename_dict["X"] of the table.
        :type rename_dict: dict

        """
        l = [i.name for i in instances.domain.attributes]
        l += [i.name for i in instances.domain.get_metas().values()]
        if instances.domain.class_var:
            l.append(instances.domain.class_var.name)
        if rename_dict is None:
            rename_dict = {}
        colList = []
        for i in l:
            colList.append(rename_dict.get(str(i), str(i)))
        try:
            cursor=self.connection.cursor()
            self.quirks.beforeWrite(cursor)
            query = 'INSERT INTO "%s" (%s) VALUES (%s);'
            for d in instances:
                valList = []
                colSList = []
                for (i, name) in enumerate(colList):
                    colSList.append('"%s"'% name)
                    valList.append(self.__attrVal2sql(d[l[i]]))
                valStr = ', '.join(["%s"]*len(colList))
                cursor.execute(query % (table,
                    ", ".join(colSList),
                    ", ".join (["%s"] * len(valList))), tuple(valList))
            cursor.close()
            self.connection.commit()
        except Exception, e:
            import traceback
            traceback.print_exc()
            self.connection.rollback()

    @deprecated_keywords({"renameDict":"rename_dict", "typeDict":"type_dict"})
    def create(self, table, instances, rename_dict = None, type_dict = None):
        """
        Create the required SQL table, then write the data into it.

        :param table: Table name
        :type table: str

        :param instances: Data to be written into the database.
        :type instances: :class:`Orange.data.Table`

        :param rename_dict: When ``rename_dict`` is provided the used names are remapped.
            The orange attribute "X" is written into the database column rename_dict["X"] of the table.
        :type rename_dict: dict

        :param type_dict: When ``type_dict`` is provided the used variables are casted into new types.
            The type of orange attribute "X" is casted into the database column of type rename_dict["X"].
        :type type_dict: dict

        """
        l = [(i.name, i.var_type ) for i in instances.domain.attributes]
        l += [(i.name, i.var_type ) for i in instances.domain.get_metas().values()]
        if instances.domain.class_var:
            l.append((instances.domain.class_var.name, instances.domain.class_var.var_type))
        if rename_dict is None:
            renameDict = {}
        colNameList = [rename_dict.get(str(i[0]), str(i[0])) for i in l]
        if type_dict is None:
            typeDict = {}
        colTypeList = [type_dict.get(str(i[0]), self.__attrType2sql(i[1])) for i in l]
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
            self.write(table, instances, rename_dict)
            self.connection.commit()
        except Exception, e:
            self.connection.rollback()

    def disconnect(self):
        """
        Disconnect from the database.
        """
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
    return r.data()