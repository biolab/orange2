DROP TABLE bus;
CREATE TABLE bus 
  (id varchar(5), 
   line integer, 
   daytime varchar, 
   temp float, 
   weather varchar, 
   arrival varchar);

LOAD DATA LOCAL INFILE 'bus.txt' INTO TABLE bus;
SELECT * from bus;
