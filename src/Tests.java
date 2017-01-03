package javaknn;
import java.util.ArrayList;

// NB: Run with -ea flag. 
public class Tests {
    public static void main(String[] args){
        testEuclid();
        testCosine();
        testClassify();

        System.out.println("TESTS OK");
    }

    public static void testEuclid() {
        Vector testA = new Vector();
        testA.putFeature(1, 5);
        testA.putFeature(2, 10);
        testA.putFeature(3, 1);
        Vector testB = new Vector();
        testB.putFeature(1, 2);
        testB.putFeature(3, 3);
        testB.putFeature(4, 2);
        
        assert(Distance.getEuclid(testA, testB) == 10.816653826391969);
    }

    public static void testCosine() {
        Vector testA = new Vector();
        testA.putFeature(1, 5);
        testA.putFeature(2, 3);
        testA.putFeature(3, 2);
        testA.putFeature(5, 2);
        Vector testB = new Vector();
        testB.putFeature(1, 3);
        testB.putFeature(2, 2);
        testB.putFeature(3, 1);
        testB.putFeature(4, 1);
        testB.putFeature(5, 1);
        testB.putFeature(6, 1);
        
        assert(Distance.getCosine(testA, testB) == 0.06439851429360033);
    }

    public static void testClassify(){
        ArrayList<Vector> train = new ArrayList<Vector>();
        Vector testA = new Vector();
        testA.putFeature(1, 5);
        testA.putFeature(2, 8);
        testA.putFeature(3, 3);
        Vector testB = new Vector();
        testB.putFeature(1, 3);
        testB.putFeature(2, 5);
        testB.setClassLabel("B");
        train.add(testB);
        Vector testC = new Vector();
        testC.putFeature(1, 1);
        testC.putFeature(2, 9);
        testC.setClassLabel("C");
        train.add(testC);
        Vector testD = new Vector();
        testD.putFeature(1, 2);
        testD.putFeature(2, 10);
        testD.putFeature(3, 1);
        testD.setClassLabel("B");
        train.add(testD);
        Vector testE = new Vector();
        testE.putFeature(1, 3);
        testE.putFeature(2, 1);
        testE.putFeature(3, 8);
        testE.setClassLabel("D");
        train.add(testE);
        Vector testF = new Vector();
        testF.putFeature(1, 5);
        testF.setClassLabel("D");
        train.add(testF);
        Vector testG = new Vector();
        testG.putFeature(1, 7);
        testG.putFeature(2, 2);
        testG.putFeature(3, 11);
        testG.setClassLabel("E");
        train.add(testG);
        Vector testH = new Vector();
        testH.putFeature(1, 9);
        testH.putFeature(4, 2);
        testH.setClassLabel("E");
        train.add(testH);
        Vector testI = new Vector();
        testI.putFeature(1, 3);
        testI.putFeature(2, 7);
        testI.putFeature(3, 1);
        testI.setClassLabel("A");
        train.add(testI);
        Vector testJ = new Vector();
        testJ.putFeature(1, 2);
        testJ.putFeature(2, 12);
        testJ.putFeature(4, 9);
        testJ.setClassLabel("E");
        train.add(testJ);
        Vector testK = new Vector();
        testK.putFeature(1, 1);
        testK.putFeature(3, 1);
        testK.putFeature(4, 5);
        testK.setClassLabel("E");
        train.add(testK);

        kNearestNeighbours.classify(testA, train, 1, "euclidian", true);
        assert(testA.getClassLabel() == "A");
        kNearestNeighbours.classify(testA, train, 2, "euclidian", true);
        assert(testA.getClassLabel() == "A");
        kNearestNeighbours.classify(testA, train, 3, "euclidian", true);
        assert(testA.getClassLabel() == "B");
        kNearestNeighbours.classify(testA, train, 4, "euclidian", true);
        assert(testA.getClassLabel() == "B");

    }
}
