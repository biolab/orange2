
import java.io.FileReader;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.lazy.BRkNN;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class MulanExp2 {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args);
        String xmlFilename = Utils.getOption("xml", args);

        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

        //RAkEL model = new RAkEL(new LabelPowerset(new NaiveBayes()));
        RAkEL model = new RAkEL(new BinaryRelevance(new NaiveBayes()));
        //RAkEL model = new RAkEL(new BRkNN(5,BRkNN.ExtensionType.EXTB));
        //RAkEL model = new RAkEL(new MLkNN(5,1.0));
        
        model.build(dataset);

        String unlabeledFilename = Utils.getOption("unlabeled", args);
        FileReader reader = new FileReader(unlabeledFilename);
        Instances unlabeledData = new Instances(reader);

        int numInstances = unlabeledData.numInstances();

        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = unlabeledData.instance(instanceIndex);
            MultiLabelOutput output = model.makePrediction(instance);
            // do necessary operations with provided prediction output, here just print it out
            System.out.printf("%d\t",instanceIndex);
            System.out.println(output);
        }
    }
}
