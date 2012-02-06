# A module that interfaces with MySQL and is used to write orange data
# tables to MySQL and read data tables from MySQL and converts them to
# Orange ExampleTable.

# Module was first drafted by Uros Rozac as a part of his
# undergraduate thesis.

# Uses:     MySQL, MySQLpy

import MySQLdb
import re, sys, string
import orange

delimit_char="$"       # delimiter for attribute name when writing to mysql table, e.g. m$meta, c$class 
delimit_char_read="$"  # delimiter used when reading attribute names mysql table
# the following names can't be used for field names; if any of these are
# used in orange, then '_' will be added to the end of the name, like select_ , or condition_ 
reserved_words=['select','condition','while','insert','update','alter','join']
char_sql="CHAR(200)"   # for length of string fields in MySQL

class Connect:
  
  def __init__(self, host, user, passwd, db):
    self.data = MySQLdb.Connect(host, user, passwd, db)
    self.SQLDomainDepot = orange.DomainDepot()
    self.special_value='""'           # for enumvariable and string
    self.special_float_value=-99999   # for int and float
    
  def load(self, table, use=None):
    data=self.query('select * from %s'%(table),use)
    return data
  
  def showTables(self):
    db=self.data
    cursor=db.cursor()
    cursor.execute('show tables')
    numrows = int(cursor.rowcount)
    print "Tables: ",
    for x in range(0,numrows-1):
      row = cursor.fetchone()
      print row[0]+",",  
    row=cursor.fetchone()
    print row[0]

  def __attrname2sql(self, a):
    pom = a.name
    subs = [("'",''), ("-",'_'), ("/",'_'), ("\\",'_'), ("+",'__'), (" ",'_')]
    for s in subs:
      pom = pom.replace(*s)
    if pom in reserved_words:            # name of attributes can't be select,insert....
      pom = "$" + pom                    # for attribute names that are SQL keywords
    if a.varType == orange.VarTypes.Continuous:
      pom += " FLOAT"
    elif a.varType == orange.VarTypes.Discrete:
      if a.values:
        pom += " ENUM (%s)" % reduce(lambda x, y: "%s, %s" % (x, y), ['"%s"' % i for i in a.values])
      else:
        pom += ' ENUM ("empty")'
    elif a.varType == orange.VarTypes.String:
      pom += " " + char_sql
    return pom

  def __attrvalue2sql(self, a):
    if a.isSpecial():
      if a.varType==orange.VarTypes.Continuous:
        return self.special_float_value
      else:
        return self.special_value
    if a.varType == orange.VarTypes.Continuous:
      return a.value
    elif a.varType == orange.VarTypes.Discrete:
      return '"%s"' % a.value
    else:
      return '"%s"' % str(a)
      
  # ************************************************************************************   
  def write(self, table, data, overwrite=0):
    db=self.data
    data_temp=data

    #Creates string with all attributes and types of attributes
    #Get Attributes
    list_variables = [self.__attrname2sql(a) for a in data.domain.attributes]
    if data.domain.classVar:
      list_variables.append("c$"+self.__attrname2sql(data.domain.classVar))

    metas = data.domain.getmetas().values()      
    list_variables.extend(["m$"+self.__attrname2sql(a) for a in metas])

    meta_exist = bool(data.domain.getmetas())

    attr_string = reduce(lambda x, y: "%s, %s" % (x, y), list_variables)
    
    #get information about tables, if table already exists...
    cursor=db.cursor()
    cursor.execute('show tables')

    for x in range(int(cursor.rowcount)):
      if table == cursor.fetchone()[0]:
        if overwrite:
          cursor.execute('drop table ' + table)
          break
        else:
          raise "Table '%s' already exists" % table

    cursor.execute('create table %s(%s)'%(table,attr_string))

    # writes data to table      
    for x in data:
      data_line = reduce(lambda x, y: ("%s, %s") % (x, y), [self.__attrvalue2sql(v) for v in x] + [self.__attrvalue2sql(x[m]) for m in metas])
      cursor.execute('insert into %s values(%s)'%(table, data_line))
      
    cursor.close()
    
  #***************************************************************************  
  def query(self, statement, use=None):
    # first, remove repeated blanks and blanks after commas
    p = re.compile('[ ]+')
    statement = p.sub(' ', statement)
    p = re.compile(', ')
    statement = p.sub(',', statement)

    db=self.data
    
    names_values=[]
    # get the name of the table, if there is one table. Otherwise table="x" (control)
    # get names of fields (with or without the names of the tables)
    stat_split=string.split(statement," ")
    manyTables = 0
    allAttributes = 0
    table=[]
    pom1_attr=[]   
    if stat_split[1]=='distinct':
      if len(string.split(stat_split[4],","))==1:
          table.append(stat_split[4])
      else:
          manyTables = 1 
          table=string.split(stat_split[4],",") #includes names of tables
      if stat_split[2]=='*':
          allAttributes = 1            # control for * in SQL statement
          for z in table:
              cursor4=db.cursor()
              cursor4.execute("describe %s" %(z))
              numrows = int(cursor4.rowcount)
              for x in range(0,numrows): 
                  row = cursor4.fetchone()
                  pom1_attr.append(row[0])    
              cursor4.close
            #control for the end of attributes in first,second...tables
              pom1_attr.append('***')           
      else:
          pom_attr=stat_split[2]
    else:
      
      if len(string.split(stat_split[3],","))==1:
            table.append(stat_split[3])
      else:
          manyTables = 1
          table=string.split(stat_split[3],",") #includes names of tables
      if stat_split[1]=='*':
          allAttributes = 1
          for z in table:
              cursor4=db.cursor()
              cursor4.execute("describe %s" %(z))
              numrows = int(cursor4.rowcount)
              for x in range(0,numrows): 
                  row = cursor4.fetchone()
                  pom1_attr.append(row[0])  
              cursor4.close
              #control for the end of attributes in first,second...tables
              pom1_attr.append('***')  
          # select statement with describe...
      else:    
          pom_attr=stat_split[1]

    if not allAttributes:
        pom1_attr=string.split(pom_attr,",")
    attr_count=len(pom1_attr)
 
    #init variables
    cursor2=db.cursor()
    table_count=0
    position_attr=-1
    hasClass = 0
    names=[]
    list_of_values = []
    attrNames = []
    intAttrs = []
    pos_int = []
    for x in pom1_attr:            # pom1_attr  'table1.field1','table2.field2' ...
      if x=='***':
        table_count += 1
      else:
        position_attr += 1
        attr=string.split(x,".")#attr 'table1','field1'
        if manyTables:
          if not allAttributes:            # there are more than one tables       
             field=attr[1]
             table_pom=attr[0]
          else:
             field=attr[0]
             table_pom=table[table_count]
        else:                              # there is only a single table
              field=attr[0]
              table_pom=table[0]

        cursor2.execute("show columns from %s like '%s' " %(table_pom,field))
        row2=cursor2.fetchone()
        pom2_values=row2[1]
        pom2_values=pom2_values[5:]
        pom2_values=pom2_values[:-1]
        
        values=string.split(row2[1],"(")  #values become second field, where the type of the field is 
        attr_type=values[0]
        list_of_values.append(pom2_values)
        attr_type=string.upper(attr_type)

        if len(field)>2 and field[1] == "$" and field[0] in "cm":
          pdAttrName = field[0]
          attrName = field[2:]
          if pdAttrName == "c":
            if hasClass:
              raise "SQL statement error: more than one class attribute specified"
            else:
              hasClass = 1
        else:
          attrName = field
          pdAttrName = ""

        if attr_type=='INT' or attr_type=='INTEGER' or attr_type=='BIGINT':
            pos_int.append(position_attr)

        if attr_type in ["ENUM", "BIT", "BOOLEAN", "BINARY"]:
          pdAttrName += "D"
        elif attr_type in ["FLOAT", "REAL", "DOUBLE"]:
          pdAttrName += "C"
        elif attr_type in ["CHAR", "TEXT", "STRING", "DATE", "TIME", "VARCHAR", "TIMESTAMP", "LONGVARCHAR"]:
          pdAttrName += "S"
        elif attr_type in ["INT", "INTEGER", "BIGINT", "SMALLINT", "DECIMAL", "TINYINT"]:
            intAttrs.append(position_attr)
            cursor3=db.cursor()
            cursor3.execute('select distinct %s from %s order by %s asc' %(field,table_pom,field))
            numrows=int(cursor3.rowcount)
            if numrows > 10:
              pdAttrName += "C"
            else:
                for z in range(0,numrows):
                   
                    row=(cursor3.fetchone())[0]
                    print row
                    row=int(row)    
                    if row<0 or row>9:
                        pdAttrName += "C"
                        break
                else:
                    pdAttrName += "D"
            cursor3.close 
        else:
            raise "cannot convert SQL values of type '%s' to Orange" % attr_type

        names.append(pdAttrName + "#" + attrName)
        attrNames.append(attrName)
        cursor2.close

    if use:
      if self.SQLDomainDepot.checkDomain(use, names):
        domain = use
      else:
        raise "the given domain does not match the query"
    else:
      domain, metaIDs, isNew = self.SQLDomainDepot.prepareDomain(names)
        
    data = orange.ExampleTable(domain)

    # create a cursor
    cursor=db.cursor()
    cursor.execute(statement)
    numrows = int(cursor.rowcount)

    # write data to example table
    for x in range(0,numrows):
      example = orange.Example(domain)
      row = cursor.fetchone()
      count_pos=0
      for i, y in enumerate(row):
         if y != self.special_float_value and y != self.special_value:
           if i in intAttrs:
              y=str(int(y))

           var = example[attrNames[i]].variable
           if type(var) == orange.EnumVariable and y not in var.values:
             var.values.append(y)
           example[attrNames[i]] = y
           
      data.append(example)

    cursor.close  
    db.close
    return data
