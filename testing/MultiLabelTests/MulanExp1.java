import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.lazy.BRkNN;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Utils;

public class MulanExp1 {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml

        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

        RAkEL learner0 = new RAkEL(new LabelPowerset(new NaiveBayes()));
        RAkEL learner1 = new RAkEL(new BinaryRelevance(new NaiveBayes()));
        RAkEL learner2 = new RAkEL(new MLkNN(5,1.0));
        RAkEL learner3 = new RAkEL(new BRkNN(5,BRkNN.ExtensionType.NONE));
        RAkEL learner4 = new RAkEL(new BRkNN(5,BRkNN.ExtensionType.EXTA));
        RAkEL learner5 = new RAkEL(new BRkNN(5,BRkNN.ExtensionType.EXTB));
        
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        int numFolds = 2;
        results = eval.crossValidate(learner0, dataset, numFolds);
        System.out.println("LabelPowerset");
        System.out.println(results);
        results = eval.crossValidate(learner1, dataset, numFolds);
        System.out.println("BinaryRelevance");
        System.out.println(results);
        results = eval.crossValidate(learner2, dataset, numFolds);
        System.out.println("MLkNN");
        System.out.println(results);
        results = eval.crossValidate(learner3, dataset, numFolds);
        System.out.println("BRkNN");
        System.out.println(results);
        results = eval.crossValidate(learner4, dataset, numFolds);
        System.out.println("BRkNN-a");
        System.out.println(results);
        results = eval.crossValidate(learner5, dataset, numFolds);
        System.out.println("BRkNN-b");
        System.out.println(results);
        
    }
}
