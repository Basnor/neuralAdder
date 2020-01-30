#include "Net.h"
#include <cstdlib>
#include <math.h>

Net::Net() {

    // Параметры сети
    static int layerSize[3] = { 10, 10, Y_NUM };
    static int trainingSet[SAMPLE_SIZE][2][3] = {
            {{0,0,0}, {0,0,1}},
            {{0,0,1}, {0,1,0}},
            {{0,1,0}, {0,1,1}},
            {{0,1,1}, {1,0,0}},
            {{1,0,0}, {1,0,1}},
            {{1,0,1}, {1,1,0}},
            {{1,1,0}, {1,1,1}}
    };


    int layerNum = sizeof(layerSize) / sizeof(int);
    m_layer = new Layer[layerNum];
    setRandNet(layerSize, layerNum);

    // Входной слой
    m_inputLayer = new Layer[1];
    m_inputLayer[0].initNeuronNum(X_NUM, 0);


    double E0 = 0.9;
    //int sumE0 = 0;
    for (int k = 0; k < 20000; k++) {

        //int i = rand() % SAMPLE_SIZE;


        for (int i = SAMPLE_SIZE-1; i >= 1; i--) {
            int j = rand() % (i + 1);

            for (int p = 0; p < 2; p++) {
                for (int t = 0; t < 3; t++) {
                    auto tmp = trainingSet[j][p][t];
                    trainingSet[j][p][t] = trainingSet[i][p][t];
                    trainingSet[i][p][t] = tmp;
                }
            }
        }

//        for (int m = 0; m < SAMPLE_SIZE; m++) {
//            std::cout << "---Входной: ";
//            for (int j = 0;j < 3; j++) {
//                std::cout << trainingSet[m][0][j];
//            }
//            std::cout << std::endl;
//        }

        int sumE0 = 0;
        for (int i = 0; i < SAMPLE_SIZE; i++) {

            setLayersValue(trainingSet[i][0], layerNum);
            E0 = getNetError(trainingSet[i][1], layerNum);
            setNetCorrection(trainingSet[i][1], layerNum);

            std::cout << " E0 " << E0 << std::endl;
            int tmp = trunc(E0*1000);

            if (tmp < 20) {
                sumE0++;
            }
        }

        std::cout << "HERE " << sumE0 << std::endl;

        if (sumE0 > SAMPLE_SIZE-1) {
            std::cout << "---НАШЕЛ!!!!!!!!!!!!" << std::endl;
            std::cout << "---k: " << k << std::endl;
            for (int m = 0; m < SAMPLE_SIZE; m++) {
                std::cout << "---Входной: ";
                for (int j = 0; j < 3; j++) {
                    std::cout << trainingSet[m][0][j];
                }
                setLayersValue(trainingSet[m][0], layerNum);
                std::cout << std::endl;


                std::cout << "---E0: " << getNetError(trainingSet[m][1], layerNum) << std::endl;


                Neuron* n = m_layer[layerNum - 1].getNeurons();
                std::cout << "---Найденный Y: " << n[0].getNeuron().value << " " << n[1].getNeuron().value << " " << n[2].getNeuron().value << std::endl;

            }

            break;

        }
    }

}

void Net::setRandNet(int* layerSize, int layerNum) {
    // Инициализируем слои со случайными значениями
    for (int i = 0; i < layerNum; i++) {
        int pastLayerSize = (i - 1 < 0) ? X_NUM : layerSize[i - 1];
        m_layer[i].initNeuronNum(layerSize[i], pastLayerSize);
        m_layer[i].setNewLayer();
    }
}

void Net::setLayersValue(int* x, int layerNum) {
    // Входной слой
    m_inputLayer[0].setInputLayer(x);

    m_layer[0].setNeuronValue(m_inputLayer[0]);

    for (int j = 1; j < layerNum; j++) {
        m_layer[j].setNeuronValue(m_layer[j - 1]);
    }
}

double Net::getNetError(int* y, int layerNum) {
    Neuron* n = m_layer[layerNum - 1].getNeurons();
    double E = 0.0;
    // Ошибка сети MSE
    for (int k = 0; k < Y_NUM; k++) {
        double inputs = (double)(y[k]);
        E += (double)pow((inputs - n[k].getNeuron().value), 2) / Y_NUM;
    }

    return E;
}

void Net::setNetCorrection(int* y, int layerNum) {
    m_layer[layerNum - 1].setTopErr(y);
    for (int k = layerNum - 2; k >= 0; k--) {
        m_layer[k].setCorrection(m_layer[k + 1].getErr(), m_layer[k + 1].getNeuronNum());
    }

    for (int k = layerNum - 1; k > 0; k--) {
        m_layer[k].setCorrectWeight(m_layer[k - 1]);
    }
    m_layer[0].setCorrectWeight(m_inputLayer[0]);
}
