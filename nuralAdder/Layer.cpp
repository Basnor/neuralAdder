#include "Layer.h"

Layer::Layer() {
    m_curNeuronNum = 0;
    m_pastNeuronNum = 0;
}

Layer::~Layer() {

}

void Layer::initNeuronNum(int curNeuronNum, int pastNeuronNum) {
    m_curNeuronNum = curNeuronNum;
    m_pastNeuronNum = pastNeuronNum;
    m_layer = new Neuron[m_curNeuronNum];
    m_err = new double[m_curNeuronNum];
}

void Layer::setNewLayer() {
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

double* Layer::getErr() {
    return m_err;
}

void Layer::setTopErr(int* req) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        double input = (double)req[i]; //output
        m_err[i] = m_layer[i].countOutputError(input);
    }
}

void Layer::setCorrection(double* nErr, int nextNeuronNum) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        m_err[i] = m_layer[i].countHidenErr(nextNeuronNum, nErr);
    }
}

void Layer::setCorrectWeight(Layer inputLayer) {
    for (int i = 0; i < m_curNeuronNum; i++) {
        m_layer[i].setWeightCorrection(m_pastNeuronNum, m_err[i], inputLayer.getNeurons());
    }
}
