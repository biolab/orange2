--orng uri 'mysql://user:somepass@localhost/test'
--orng discrete ['registration', 'num', 'time of day', 'arrival']
--orng meta ['weather', 'arrival', 'time']
--orng class ['arrival']

SELECT
    "id" as registration,
    line as num,
    daytime as "time of day",
    temp as temperature,
    weather,
    arrival
FROM 
    bus
WHERE 
    line='10';