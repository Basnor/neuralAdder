#ifndef NEURON_H
#define NEURON_H

class Neuron
{
public:
    Neuron();
    ~Neuron();

    struct neuron
    {
        double value;     // значение
        double* weight;   // веса
        double bias;      // смещение
    };

    void setRandNeuron(int waightsNum);
    void setNeuronValue(double val);
    void setValues(Neuron* pastLayer, int pastLayerSize);

    double countOutputError(double req);
    double countHidenErr(int waightsNum, double* outError);
    void setWeightCorrection(int waightsNum, double error, Neuron* pastLayer);

    neuron getNeuron();

private:
    neuron m_neuron;
};

#endif

