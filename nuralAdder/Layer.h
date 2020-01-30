
#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include "Neuron.h"

class Layer
{
public:
    Layer();
    ~Layer();

    void initNeuronNum(int curNeuronNum, int pastNeuronNum);
    void setNewLayer();
    void setInputLayer(int* values);
    void setNeuronValue(Layer inputLayer);
    void setCorrection(double* nErr, int nextNeuronNum);
    void setCorrectWeight(Layer inputLayer);
    double* getErr();
    void setTopErr(int* req);

    int getNeuronNum();
    Neuron* getNeurons();

private:
    Neuron* m_layer;
    double* m_err;
    int m_curNeuronNum;
    int m_pastNeuronNum;
};

#endif
