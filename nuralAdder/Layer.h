#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

class Layer
{
public:
    /*
    * Constructor
    */
    Layer();

    /*
    * Init size and composition of the layer
    */
    void initNeuronNum(int curNeuronNum, int pastNeuronNum);

    /*
    * Fill neurons with random values
    */
    void setRandNeurons();

    /*
    * Set input layer values
    */
    void setInputLayer(int* values);

    /*
    * Set neuron valuses for hiden layers
    */
    void setNeuronValue(Layer inputLayer);

    /*
    * Set error for neurons in hiden layers
    */
    void setHidenLayersErrors(Neuron* nextNeurons, int nextNeuronNum);

    /*
    * Set correct weight and bias for neurons in current layer
    */
    void setNeuronsCorrection(Layer inputLayer);

    /*
    * Set error for neurons in output layer
    */
    void setOutputLayerErrors(int* req);

    /*
    * Get number of neurons in current layer
    */
    int getNeuronNum();

    /*
    * Cet neurons in current layer
    */
    Neuron* getNeurons();

private:
    /*
    * Neurons in current layer
    */
    Neuron* m_layer;

    /*
    * Number neurons in current layer
    */
    int m_curNeuronNum;

    /*
    * Number neurons in previous layer
    */
    int m_pastNeuronNum;
};

#endif
