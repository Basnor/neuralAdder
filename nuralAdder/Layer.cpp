#include "Layer.h"

Layer::Layer() {
    m_curNeuronNum = 0;
    m_pastNeuronNum = 0;
}

void Layer::initNeuronNum(int curNeuronNum, int pastNeuronNum) {
    m_curNeuronNum = curNeuronNum;
    m_pastNeuronNum = pastNeuronNum;
    m_layer = new Neuron[m_curNeuronNum];
}

void Layer::setRandNeurons() {
    for (int i = 0; i < m_curNeuronNum; i++) {
        m_layer[i].setRandNeuron(m_pastNeuronNum);
    }
}

void Layer::setInputLayer(int* values) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        double input = (double)values[i];
        m_layer[i].setNeuronValue(input);
    }
}

void Layer::setNeuronValue(Layer inputLayer) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        m_layer[i].setValues(inputLayer.getNeurons(), inputLayer.getNeuronNum());
    }
}

int Layer::getNeuronNum() {
    return m_curNeuronNum;
}

Neuron* Layer::getNeurons() {
    return m_layer;
}

void Layer::setOutputLayerErrors(int* req) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        double reqValue = (double)req[i];
        m_layer[i].setOutputError(reqValue);
    }
}

void Layer::setHidenLayersErrors(Neuron* nextNeurons, int nextNeuronNum) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        m_layer[i].setHidenErr(nextNeuronNum, nextNeurons);
    }
}

void Layer::setNeuronsCorrection(Layer pastLayer) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        m_layer[i].setNeuronCorrection(m_pastNeuronNum, pastLayer.getNeurons());
    }
}
