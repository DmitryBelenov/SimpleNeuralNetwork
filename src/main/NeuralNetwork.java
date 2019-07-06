package main;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

/**
 * Нейронная сеть (feedforward, back propagation)
 * <p>
 * Параметры сети:
 * learningR - коэффициент обучаемости
 * (чем выше, тем быстрее обучается сеть. при этом снижается качество)
 * hL1Size - размер первого скрытого слоя
 * hL2Size - размер второго скрытого слоя
 * expected - ожидаемый результат для определенного кейса
 * <p>
 * Обучение и проверка предсказания происходят в одной сессии,
 * хранимой памяти у сети нет
 *
 * @author Dmitry Belenov
 */

public class NeuralNetwork {
    private static final double learningR = 0.05;
    //randomize
    private static final double rangeMin = -1.0;
    private static final double rangeMax = 1.0;

    private static double expected;
    private static double[] input;
    private static double error;
    //hidden layers size
    private static int hL1Size = 150;
    private static int hL2Size = 50;

    //network condition
    private static double[][] hl1Wghts;
    private static Map<Double, double[]> hiddenLayerMap1 = new HashMap<>();
    private static double[] hl1Errors;
    private static double[] hiddenLayer1;

    private static double[][] hl2Wghts;
    private static Map<Double, double[]> hiddenLayerMap2 = new HashMap<>();
    private static double[] hl2Errors;
    private static double[] hiddenLayer2;

    private static double[] resWghts;
    private static Map<Double, double[]> resultMap = new HashMap<>();
    private static double result;

    private static int i = 0;
    private static int process = 0;

    private static double rndDouble() {
        Random random = new Random();
        return rangeMin + (rangeMax - rangeMin) * random.nextDouble();
    }

    // activation function (e ~ 2,71828)
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // x - sigmoid of num
    private static double sigmoidDx(double x) {
        return x * (1 - x);
    }

    private static double doubleScale(double x) {
        return new BigDecimal(x).setScale(3, RoundingMode.DOWN).doubleValue();
    }

    private static double[] synapses(int size) {
        double[] X = new double[size];
        for (int i = 0; i < size; i++) {
            double rnd = rndDouble();
            X[i] = rnd;
        }
        return X;
    }

    //neuron structure
    private static double neuron(double[] input,
                                 int layerNum,
                                 boolean back,
                                 double[] synapses) {
        int size = input.length;

        double[] weights = back ? synapses : synapses(size);
        double[] sums = new double[size];

        for (int i = 0; i < size; i++) {
            double oneRes = input[i] * weights[i];
            sums[i] = oneRes;
        }
        //summer
        double s = 0;
        for (double sum : sums) {
            s += sum;
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
    private static double initialize(double[] in) {
        input = in;
        hiddenLayer1 = new double[hL1Size];
        for (int i = 0; i < hL1Size; i++) {
            hiddenLayer1[i] = neuron(in, 1, false, null);
        }

        hiddenLayer2 = new double[hL2Size];
        for (int j = 0; j < hL2Size; j++) {
            hiddenLayer2[j] = neuron(hiddenLayer1, 2, false, null);
        }

        result = neuron(hiddenLayer2, 3, false, null);
        error = result - expected;

        return result;
    }

    private static double[] renewedWeights(double neuron,
                                           double error,
                                           double[] beforeLayer,
                                           Map<Double, double[]> currentLayerMap,
                                           int layerNum) {
        int size = beforeLayer.length;
        double[] weights = currentLayerMap.get(neuron);
        double[] newWeights = new double[size];
        double[] errors = new double[size];

        double delta = error * sigmoidDx(neuron);
        for (int i = 0; i < size; i++) {
            newWeights[i] = weights[i] - (beforeLayer[i] * delta * learningR);
            errors[i] = newWeights[i] * delta;
        }

        if (layerNum == 3) {
            hl2Errors = errors;
        } else if (layerNum == 2) {
            hl1Errors = errors;
        }
        return newWeights;
    }

    //out layer weights renew
    private static void backOut() {
        resWghts = renewedWeights(result, error, hiddenLayer2, resultMap, 3);
    }

    //back propagation
    private static void backPropagation() {
        int size = hiddenLayer2.length;
        hl2Wghts = new double[size][];

        double[] betweenErr = null;
        for (int i = 0; i < size; i++) {
            double[] w = renewedWeights(hiddenLayer2[i], hl2Errors[i], hiddenLayer1, hiddenLayerMap2, 2);
            hl2Wghts[i] = w;

            for (int j = 0; j < hl1Errors.length; j++) {
                if (betweenErr != null) {
                    hl1Errors[j] += betweenErr[j];
                }
            }
            betweenErr = hl1Errors;
        }

        //sigmoid from errors
        for (int i = 0; i < hl1Errors.length; i++) {
            hl1Errors[i] = sigmoid(hl1Errors[i]);
        }

        size = hiddenLayer1.length;
        hl1Wghts = new double[size][];

        for (int i = 0; i < size; i++) {
            double[] w = renewedWeights(hiddenLayer1[i], hl1Errors[i], input, hiddenLayerMap1, 0); //layer 0 - input
            hl1Wghts[i] = w;
        }
    }

    private static double newForward(double[] in) {
        hiddenLayer1 = new double[hL1Size];
        for (int i = 0; i < hL1Size; i++) {
            hiddenLayer1[i] = neuron(in, 1, true, hl1Wghts[i]);
        }

        hiddenLayer2 = new double[hL2Size];
        for (int j = 0; j < hL2Size; j++) {
            hiddenLayer2[j] = neuron(hiddenLayer1, 2, true, hl2Wghts[j]);
        }

        result = neuron(hiddenLayer2, 3, true, resWghts);
        error = result - expected;

        return result;
    }

    private static void learn(double[] input, boolean init) {
        if (init) {
            initialize(input);
        } else {
            newForward(input);
        }

        double x = Math.abs(error);
        while (x > 0.01) {
            backOut();
            backPropagation();
            newForward(input);
            x = Math.abs(error);
            i++;
            progress();
        }
    }

    private static void progress() {
        process++;
        if (process == 200) {
            System.out.print(">");
            process = 0;
        }
    }

    private static void checkCase() {
        double[] test1 = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expected = 1;
        learn(test1, true);

        double[] test2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expected = 0;
        learn(test2, false);

        double[] test3 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expected = 1;
        learn(test3, false);

        double[] test4 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
        expected = 0;
        learn(test4, false);

        System.out.println("> COMPLETE!\n");
        System.out.println("ITERATIONS: " + i + "\n");

        System.out.print("PRESS ENTER:");
        Scanner sc4 = new Scanner(System.in);
        String s4 = sc4.nextLine();

        // - указать кейс для предсказания
        double[] check = {0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        newForward(check);
        double res = doubleScale(result);
        System.out.println("\n+- - - - - +- - - - - - +");
        System.out.println("| RESULT   | " + (res > 0.1 ? "1" : "0") + " - " + res * 100 + "%");
        System.out.println("+- - - - - +- - - - - - +");
    }

    public static void main(String[] args) {
        checkCase();
    }
}
