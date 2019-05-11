package main;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class NeuralNetwork {

    /**
     * Нейронная сеть (feedforward, back propagation)
     *
     * Параметры сети:
     * при обучении ожидаемые 0 или 1
     * вход {max = 6}
     * обучающих пар {max = 3}
     * learning rate = 0.01
     * 1й скрытый слой {10}
     * 2й скрытый слой {5}
     *
     * Обучение и проверка предсказания происходят в одной сессии,
     * хранимой памяти у сети нет
     *
     * ВАЖНО:
     * При проверке результатов нужно учесть последнее значение 'expected':
     * условие, когда в обучении последний 'expected' был "0":
     * <0.1 = 0, >0.1 = 1
     * условие, когда в обучении последний 'expected' был "1":
     * >0.9 = 1, <0.9 = 0
     *
     *
     * @author Dmitry Belenov
     *
     * */

    private static double expected;
    private static final double learningR = 0.01;
    private static double[] input;
    private static double error;

    //randomize
    private static final double rangeMin = -1.0;
    private static final double rangeMax = 1.0;

    //hidden layers size
    private static int hL1Size = 10;
    private static int hL2Size = 5;

    //network condition
    private static double [][] hl1Wghts;
    private static Map<Double, double []> hiddenLayerMap1 = new HashMap<>();
    private static double [] hl1Errors;
    private static double [] hiddenLayer1;

    private static double [][] hl2Wghts;
    private static Map<Double, double []> hiddenLayerMap2 = new HashMap<>();
    private static double [] hl2Errors;
    private static double [] hiddenLayer2;

    private static double [] resWghts;
    private static Map<Double, double []> resultMap = new HashMap<>();
    private static double result;

    private static double rndDouble () {
        Random random = new Random();
        return rangeMin + (rangeMax - rangeMin) * random.nextDouble();
    }

    // activation function (e ~ 2,71828)
    private static double sigmoid (double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // x - sigmoid of num
    public static double sigmoidDx (double x) { return x * (1 - x);
    }

    private static double doubleScale (double x) {
        double nd = new BigDecimal(x).setScale(5, RoundingMode.DOWN).doubleValue();
        return nd;
    }

    private static double[] synapses (int size) {
        double[] X = new double[size];
        for (int i=0; i<size; i++) {
            double rnd = rndDouble();
            X[i] = rnd;
        }
        return X;
    }

    //neuron structure
    public static double neuron (double [] input,
                                 int layerNum,
                                 boolean back,
                                 double[] synapses) {
        int size = input.length;

        double [] weights = back ? synapses : synapses(size);
        double [] sums = new double[size];

        for (int i=0; i<size; i++) {
            double oneRes = input[i] * weights[i];
            sums[i]=oneRes;
        }
        //summer
        double s = 0;
        for (int j=0; j<sums.length; j++) {
            s += sums[j];
        }
        //activator
        double res = sigmoid(s);

        if (layerNum == 3) {
            resultMap.put(res, weights);
        } else {
            if (layerNum == 1) {
                hiddenLayerMap1.put(res, weights);
            } else {
                hiddenLayerMap2.put(res, weights);
            }
        }
        return res;
    }

    //initialize
    private static double initialize (double [] in) {
        input = in;
        hiddenLayer1 = new double[hL1Size];
        for (int i=0; i<hL1Size; i++) {
            hiddenLayer1[i]= neuron(in, 1, false, null);
        }

        hiddenLayer2 = new double[hL2Size];
        for (int j=0; j<hL2Size; j++) {
            hiddenLayer2[j]= neuron(hiddenLayer1, 2,false, null);
        }

        result = neuron(hiddenLayer2, 3,false, null);
        error = result - expected;

        return result;
    }

    private static double[] renewedWeights (double neuron,
                                            double error,
                                            double[] beforeLayer,
                                            Map<Double, double[]> currentLayerMap,
                                            int layerNum) {
        int size = beforeLayer.length;
        double[] weights = currentLayerMap.get(neuron);
        double[] newWeights = new double[size];
        double[] errors = new double[size];

        double delta = error * sigmoidDx(neuron);
        for (int i=0; i<size; i++) {
            newWeights[i] = weights[i] - (beforeLayer[i] * delta * learningR);
            errors[i] = newWeights[i] * delta;
        }

        if (layerNum == 3) {
            hl2Errors = errors;
        } else
        if (layerNum == 2) {
            hl1Errors = errors;
        }
        return newWeights;
    }

    //out layer weights renew
    private static void backOut() {
        resWghts = renewedWeights(result, error, hiddenLayer2, resultMap, 3);
    }

    //back propagation
    private static void backPropagation () {
        int size = hiddenLayer2.length;
        hl2Wghts = new double[size][];

        double[] betweenErr = null;
        for (int i=0; i<size; i++) {
            double[] w = renewedWeights(hiddenLayer2[i],hl2Errors[i],hiddenLayer1,hiddenLayerMap2,2);
            hl2Wghts[i] = w;

            for (int j=0; j<hl1Errors.length; j++) {
                if (betweenErr != null) {
                    hl1Errors[j] += betweenErr[j];
                }
            }
            betweenErr = hl1Errors;
        }

        //sigmoid from errors
        for (int i=0; i<hl1Errors.length; i++) {
            hl1Errors[i] = sigmoid(hl1Errors[i]);
        }

        size = hiddenLayer1.length;
        hl1Wghts = new double[size][];

        for (int i=0; i<size; i++) {
            double[] w = renewedWeights(hiddenLayer1[i],hl1Errors[i],input,hiddenLayerMap1,0); //layer 0 - input
            hl1Wghts[i] = w;
        }
    }

    private static double newForward (double [] in) {
        hiddenLayer1 = new double[hL1Size];
        for (int i=0; i<hL1Size; i++) {
            hiddenLayer1[i]= neuron(in, 1, true, hl1Wghts[i]);
        }

        hiddenLayer2 = new double[hL2Size];
        for (int j=0; j<hL2Size; j++) {
            hiddenLayer2[j]= neuron(hiddenLayer1, 2, true, hl2Wghts[j]);
        }

        result = neuron(hiddenLayer2, 3, true, resWghts);
        error = result - expected;

        print();
        return result;
    }

    private static void learn_init(double[] input){
        int i = 0;
        initialize(input);
        print();

        double x = Math.abs(error);
        while (x > 0.01) {
            backOut();
            backPropagation();
            newForward(input);
            x = Math.abs(error);
            i++;
        }

        System.out.println("\n\nITERATIONS: "+i);
    }

    private static void learn (double[] input){
        int i = 0;
        newForward(input);

        double x = Math.abs(error);
        while (x > 0.01) {
            backOut();
            backPropagation();
            newForward(input);
            x = Math.abs(error);
            i++;
        }

        System.out.println("\n\nITERATIONS: "+i);
    }

    private static void print () {
        System.out.print("\n\nhidden layer1: ");
        for (int i=0; i<hiddenLayer1.length; i++) {
            System.out.print(doubleScale(hiddenLayer1[i])+" | ");
        }
        System.out.print("\nhidden layer2: ");
        for (int i=0; i<hiddenLayer2.length; i++) {
            System.out.print(doubleScale(hiddenLayer2[i])+" | ");
        }
    }

    private static void testCase() {
        //указать кейсы для обучения (не более 3х, максимальный размер массива - 6)
        double [] test1 = {1,1,1,0,0,0};
        expected = 1;
        learn_init(test1);

        double [] test2 = {0,0,0,1,1,1};
        expected = 0; //последний ожидаемый результат влияет на определение предсказания сети (см.комментарий к классу)
        learn(test2);

        //добавить 3й кейс при необходимости
//        double [] test3 = {1,1,1,1,1,1};
//        expected = 1;
//        learn(test3);

        System.out.println("+- - - - - - - - - - -+- - - - - - - - - - -+- - - - - - - - - - -+");
        System.out.println("|                     | LET'S CHECK RESULTS |                     |");
        System.out.println("+- - - - - - - - - - -+- - - - - - - - - - -+- - - - - - - - - - -+");

        System.out.print("\nPRESS ENTER:");
        Scanner sc4 = new Scanner(System.in);
        String s4 = sc4.nextLine();

        double [] check = {0,0,1,0,0,0}; // - указать кейс для предсказания
        newForward(check);
        System.out.println("\n+- - - - - +- - - - - +");
        System.out.println("| RESULT   | "+doubleScale(result));
        System.out.println("+- - - - - +- - - - - +");
    }

    public static void main(String[] args){
        testCase();
    }
}
