Data Table
==========

.. image:: DataTable_icon.png
   :alt: Data Table icon
  
Signals
-------

Inputs:
    - Examples (ExampleTable)
        Attribute-valued data set.
      
Outputs:
    - Selected Examples (Example Table)
        Selected data instalces
        
Description
-----------
    
Data Table widget takes one or more data sets on its input, and presents
them in a spreadsheet format. Widget supports sorting by attribute 
values (click on the attribute name in the header row). 


Examples
--------

We used two `File` widgets, read the iris and glass data set (provided in Orange distribution), and send them to the Data Table widget.

.. image:: DataTable_schema.*
   :alt: Example data table schema
   
A snapshot of the widget under these settings is shown below.

.. image::  DataTable.*
   :alt: bla