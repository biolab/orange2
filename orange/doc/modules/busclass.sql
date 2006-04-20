USE test;
DROP TABLE busclass;
CREATE TABLE busclass 
  (m$id varchar(5), 
   line enum('9','10','11'), 
   daytime enum('morning','evening', 'midday'), 
   temp float, weather enum('rainy','sunny'), 
   c$arrival enum('late','on-time'));
LOAD DATA LOCAL INFILE 'bus.txt' INTO TABLE busclass;

SELECT * FROM busclass;
