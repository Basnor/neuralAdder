#include "Neuron.h"
#include <math.h>

Neuron::Neuron(double learningRate) {
    m_learningRate = learningRate;
}

Neuron::neuron Neuron::getNeuron() {
    return m_neuron;
}

void Neuron::setNeuronValue(double val) {
    m_neuron.value = val;
}

void Neuron::setRandNeuron(int waightsNum) {
    m_neuron.weights = new double[waightsNum];
    for (int i = 0; i < waightsNum; i++) {
        m_neuron.weights[i] = getRandVal();
    }

    m_neuron.bias = getRandVal();
}

double Neuron::getRandVal(){
    // Rand value: [-0.500; -0.010] v [0.010; 0.500]
    double val = (double)(10 + rand() % 490) / 1000.0;
    if (rand() % 2) {
        val *= -1;
    }

    return val;
}

void Neuron::setValues(Neuron* pastLayer, int pastLayerSize) {
    // Sigmoid Activation Function
    double ex = m_neuron.bias;
    for (int i = 0; i < pastLayerSize; i++) {
        ex += pastLayer[i].getNeuron().value * m_neuron.weights[i];
    }

    m_neuron.value = 1.0 / (1.0 + exp(-1.0 * ex));
}

void Neuron::setOutputError(double req) {
    m_neuron.err = m_neuron.value * (1.0 - m_neuron.value) * 2.0 /
            3.0 * (m_neuron.value - req);
}

void Neuron::setHidenErr(int nextLayerSize, Neuron* nextNeurons) {
    m_neuron.err = 0.0;

    double k = m_neuron.value * (1.0 - m_neuron.value);
    for (int i = 0; i < nextLayerSize; i++) {
        m_neuron.err += k * nextNeurons[i].getNeuron().err * m_neuron.weights[i];
    }
}

void Neuron::setNeuronCorrection(int pastLayerSize, Neuron* pastLayer) {
    double deltaB = -1.0 * m_learningRate * m_neuron.err * 1.0;
    m_neuron.bias += deltaB;

    for (int i = 0; i < pastLayerSize; i++) {
        double deltaW = -1.0 * m_learningRate * m_neuron.err * pastLayer[i].getNeuron().value;
        m_neuron.weights[i] += deltaW;
    }
}
