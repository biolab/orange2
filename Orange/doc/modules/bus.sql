--DROP TABLE bus;
CREATE TABLE bus 
  (id varchar(5), 
   line enum('9','10','11'), 
   daytime enum('morning','evening', 'midday'), 
   temp float, 
   weather enum('rainy','sunny'), 
   arrival enum('late','on-time'));
LOAD DATA LOCAL INFILE 'bus.txt' INTO TABLE bus;

SELECT * FROM bus;
