##################################
SQL interface (``sql``)
##################################

The :class:`sql` module provides access to relational databases from Orange.
It currently supports:

- `MySql <http://www.mysql.com/>`_  through `MySQL for Python <http://sourceforge.net/projects/mysql-python/>`_,
- `Postgres <http://www.postgresql.org>`_ through `Psycopg <http://initd.org/psycopg/>`_,
- `sqlite <http://www.sqlite.org/>`_ through `sqlite3 <http://docs.python.org/library/sqlite3.html>`_.

:class:`SQLReader` and :class:`SQLWriter` classes require connection string based on
standard format scheme://[user[:password]@]host[:port]/database[?parameters].

Examples of valid connection strings:

- sqlite://database.db/
- mysql://user:password@host/database
- mysql://host/database?debug=1
- postgres://user@host/database?debug=&cache=
- postgres://host:5432/database

Attribute Names and Types
-------------------------

Rows returned by an SQL query have to be converted into Orange examples.
Each column in a row has to be converted into a certain feature type. The
following conversions between SQL and Orange types are used:

- STRING and DATETIME attributes are converted to Orange strings.

- The features listed in ``discrete_names`` are converted to Orange
  discrete features.

- Other features are converted to continuous Orange features.

- The attribute in ``class_name`` is set as the class features. If no
  ``class_name`` is set, the column with the name "class" in the
  returned SQL query is set as the class attribute. If no such column
  exists, the last column is set as the class features.

.. note:: When reading ``sqlite`` data table into :class:`Orange.data.Table` all columns are cast into :class:`Orange.feature.String`.

**Examples**

The following example populates the `sqlite <http://www.sqlite.org/>`_ database with data from :class:`Orange.data.Table`.

.. literalinclude:: code/sql-example.py
   :lines: 1-6

Using the existing `sqlite <http://www.sqlite.org/>`_ database one can fetch back the data into :class:`Orange.data.Table`.

.. literalinclude:: code/sql-example.py
   :lines: 8-24

The output of the last example is::

  150 instances returned
  Output data domain:
  StringVariable 'sepal length'
  StringVariable 'sepal width'
  StringVariable 'petal length'
  StringVariable 'petal width'
  StringVariable 'iris'
  First instance : ['5.09999990463', '3.5', '1.39999997616', '0.20000000298', 'Iris-setosa']
  22 instances returned

.. autoclass:: Orange.data.sql.SQLReader
   :members:

.. autoclass:: Orange.data.sql.SQLWriter
   :members: