#include "Net.h"
#include <cstdlib>
#include <math.h>

Net::Net() {

    // Neuron num in hiden layers
    static int HidenLayersStruct[3] = { 10, 10, Y_NUM };
    static int trainingSet[SAMPLE_SIZE][2][3] = {
            {{0,0,0}, {0,0,1}},
            {{0,0,1}, {0,1,0}},
            {{0,1,0}, {0,1,1}},
            {{0,1,1}, {1,0,0}},
            {{1,0,0}, {1,0,1}},
            {{1,0,1}, {1,1,0}},
            {{1,1,0}, {1,1,1}}
    };

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 3; k++) {
                m_trainingSet[i][j][k] = trainingSet[i][j][k];
            }
        }
    }

    // Hiden and output layers
    m_layersNum = sizeof(HidenLayersStruct) / sizeof(int);
    m_layer = new Layer[m_layersNum];
    setRandNetParam(HidenLayersStruct, m_layersNum);

    // Input layer
    m_inputLayer = new Layer;
    m_inputLayer->initNeuronNum(X_NUM, 0);

}

void Net::trainNet(){
    for (int i = 0; i < MAX_EPOCH; i++) {
        shuffleTrainingSet();
        correctNetOnSet();

        if (isNetTrained()) {
            std::cout << "Training success." << std::endl;
            std::cout << "Epoch num: " << i << std::endl;
            break;
        }

        if (i == MAX_EPOCH - 1) {
            std::cout << "Retraining. Try again." << std::endl;
        }
    }

}

void Net::correctNetOnSet(){
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        setLayersValues(m_trainingSet[i][0], m_layersNum);
        setNetCorrection(m_trainingSet[i][1], m_layersNum);
    }
}

bool Net::isNetTrained(){
    int smallErrorNum = 0;
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        setLayersValues(m_trainingSet[i][0], m_layersNum);
        double outputError = getNetError(m_trainingSet[i][1], m_layersNum);

        // Error < 0.02
        int error = trunc( outputError*1000 );
        if (error < 20) {
            smallErrorNum++;
        }
    }

    if (smallErrorNum > SAMPLE_SIZE - 1) {
        return true;
    }

    return false;
}

void Net::setRandNetParam(int* layerSize, int layerNum) {
    for (int i = 0; i < layerNum; i++) {
        int pastLayerSize = (i - 1 < 0) ? X_NUM : layerSize[i - 1];
        m_layer[i].initNeuronNum(layerSize[i], pastLayerSize);
        m_layer[i].setNewLayer();
    }
}

void Net::setLayersValues(int* x, int layerNum) {
    // Input layer
    m_inputLayer[0].setInputLayer(x);

    m_layer[0].setNeuronValue(m_inputLayer[0]);

    for (int j = 1; j < layerNum; j++) {
        m_layer[j].setNeuronValue(m_layer[j - 1]);
    }
}

double Net::getNetError(int* y, int layerNum) {
    Neuron* n = m_layer[layerNum - 1].getNeurons();
    double E = 0.0;
    // MSE
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

void Net::shuffleTrainingSet(){
    for (int i = SAMPLE_SIZE-1; i >= 1; i--) {
        int j = rand() % (i + 1);

        for (int p = 0; p < 2; p++) {
            for (int t = 0; t < 3; t++) {
                int tmp = m_trainingSet[j][p][t];
                m_trainingSet[j][p][t] = m_trainingSet[i][p][t];
                m_trainingSet[i][p][t] = tmp;
            }
        }
    }
}

void Net::showResultsForAllSets(){
    std::cout << "Result: " << std::endl;

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        std::cout << "---Input: ";
        for (int j = 0; j < 3; j++) {
            std::cout << m_trainingSet[i][0][j];
        }
        setLayersValues(m_trainingSet[i][0], m_layersNum);
        std::cout << std::endl;

        std::cout << "---Error: " << getNetError(m_trainingSet[i][1], m_layersNum) << std::endl;

        Neuron* n = m_layer[m_layersNum - 1].getNeurons();
        std::cout << "---Output: " << n[0].getNeuron().value << " " << n[1].getNeuron().value << " " << n[2].getNeuron().value << std::endl;

    }
}
