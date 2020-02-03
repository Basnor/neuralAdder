#ifndef NEURON_H
#define NEURON_H

class Neuron
{
    public:
    /*
    * Constructor
    */
    Neuron(double learningRate = 0.7);

    struct neuron
    {
        double value;
        double* weights;
        double bias;
        double err;
    };

    /*
    * Set rand weights and bias for m_neuron
    */
    void setRandNeuron(int waightsNum);

    /*
    * Set recieved neuron value
    */
    void setNeuronValue(double val);

    /*
    * Sigmoid Activation Function
    */
    void setValues(Neuron* pastLayer, int pastLayerSize);

    /*
    * set error of last layer
    */
    void setOutputError(double req);

    /*
    * Errors of hiden layers
    */
    void setHidenErr(int nextLayerSize, Neuron* nextNeurons);

    /*
    * Weight and Bias correction for current neuron
    * waightsNum - number of weights included in current neuron
    * pastLayer - neurons params of previous layer
    */
    void setNeuronCorrection(int pastLayerSize, Neuron* pastLayer);

    /*
    * Get current neuron params
    */
    neuron getNeuron();

private:
    double m_learningRate;

    /*
    * Current neuron
    */
    neuron m_neuron;

    /*
    * Get rand value: [-0.500; -0.010] v [0.010; 0.500]
    */
    double getRandVal();

};

#endif

