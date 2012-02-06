DROP TABLE bus;
CREATE TABLE bus 
  (id varchar(5), 
   line integer, 
   daytime varchar, 
   temp float, 
   weather varchar, 
   arrival varchar);

\COPY "bus" FROM 'bus.txt' USING DELIMITERS '	'
SELECT * from bus;