##################################
SQL interface (``sql``)
##################################

The :class:`sql` module provides access to relational databases from Orange.
It currently supports `MySql <http://www.mysql.com/>`_  through
`MySQL for Python <http://sourceforge.net/projects/mysql-python/>`_
and `Postgres <http://www.postgresql.org>`_ through `Psycopg <http://initd.org/psycopg/>`_.


:class:`SQLReader` and :class:`SQLWriter` classes require connection string based on
standard format scheme://[user[:password]@]host[:port]/database[?parameters].

Examples of valid connection strings:

- mysql://user:password@host/database
- mysql://host/database?debug=1
- postgres://user@host/database?debug=&cache=
- postgres:///full/path/to/socket/database
- postgres://host:5432/database

**Attribute Names and Types**

Rows returned by an SQL query have to be converted into Orange Instances.
Each column in a row has to be converted into a certain feature type. The
following conversions between SQL and Orange types are used:

- STRING and DATETIME attributes are converted to Orange strings.
- The features listed in ``discrete_names`` are converted to Orange discrete features.
- Other features are converted to continuous Orange features.
- The attribute in ``class_name`` is set as the class features. If no ``class_name`` is set, the column with the
  name "class" in the returned SQL query is set as the class attribute. If no such column exists, the last
  column is set as the class features.

.. autoclass:: Orange.data.sql.SQLReader
   :members:

.. autoclass:: Orange.data.sql.SQLWriter
   :members: