javac -cp mulan.jar:weka.jar MulanExp2.java 
java -cp mulan.jar:weka.jar:. MulanExp2 -arff emotions.arff -xml emotions.xml -unlabeled emotions.arff >exp_out.txt
