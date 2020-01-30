#include "Neuron.h"
#include <iostream>
#include <cstdlib>
#include <math.h>

Neuron::Neuron() {

}

Neuron::~Neuron() {

}

Neuron::neuron Neuron::getNeuron() {
    return m_neuron;
}

void Neuron::setNeuronValue(double val) {
    m_neuron.value = val;
}

void Neuron::setRandNeuron(int waightsNum) {
    m_neuron.weight = new double[waightsNum];

    // Веса [-0.100; -0.010] v [0.010; 0.100]
    for (int i = 0; i < waightsNum; i++) {
        int r = 10 + rand() % 490;//90
        m_neuron.weight[i] = (double)(r) / 1000.0;
        int isMinus = rand() % 2;
        if (isMinus) {
            m_neuron.weight[i] *= -1;
        }
    }

    // Смещение
    int r = 10 + rand() % 490;//90
    m_neuron.bias = (double)(r) / 1000.0;
    int isMinus = rand() % 2;
    if (isMinus) {
        m_neuron.bias *= -1;
    }
}

void Neuron::setValues(Neuron* pastLayer, int pastLayerSize) {
    // Лог-сигмоидная функция активации нейронов
    double ex = m_neuron.bias;
    for (int i = 0; i < pastLayerSize; i++) {
        ex += pastLayer[i].getNeuron().value * m_neuron.weight[i];
    }

    m_neuron.value = 1.0 / (1.0 + exp(-1.0 * ex));
}

double Neuron::countOutputError(double req) {
//    double er = m_neuron.value * (1.0 - m_neuron.value) * (m_neuron.value - req);
    double er = m_neuron.value * (1.0 - m_neuron.value) * 2.0 / 3.0 * (m_neuron.value - req);
    return er;
}

double Neuron::countHidenErr(int waightsNum, double* outError) {
    double er = 0.0;

    double k = m_neuron.value * (1.0 - m_neuron.value);
    for (int i = 0; i < waightsNum; i++) {
        er += k * outError[i] * m_neuron.weight[i];
    }
    return er;
}

void Neuron::setWeightCorrection(int waightsNum, double error, Neuron* pastLayer) {
    double deltaB = -0.7 * error * 1.0;
    m_neuron.bias += deltaB;

    for (int i = 0; i < waightsNum; i++) {
        double deltaW = -0.7 * error * pastLayer[i].getNeuron().value;
        m_neuron.weight[i] += deltaW;
    }
}
