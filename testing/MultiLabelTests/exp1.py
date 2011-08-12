javac -cp mulan.jar:weka.jar MulanExp1.java 
java -cp mulan.jar:weka.jar:. MulanExp1 -arff emotions.arff -xml emotions.xml >eval.txt
