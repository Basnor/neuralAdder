#ifndef NET_H
#define NET_H

#include "Layer.h"

class Net
{
public:
    enum { X_NUM = 3, Y_NUM = 3, SAMPLE_SIZE = 7, MAX_EPOCH = 20000 };

    /*
    * Constructor
    */
    Net();

    /*
    * Train net on sample sets
    */
    void trainNet();

    /*
    * Show net errors and net values for all training sets
    */
    void showResultsForAllSets();

private:
    int m_trainingSet[SAMPLE_SIZE][2][3];

    /*
    * Num of hiden + output (1) layers
    */
    int m_layersNum;

    /*
    * Layers exept first (input)
    */
    Layer* m_layer;

    /*
    * Input layer
    */
    Layer* m_inputLayer;

    /*
    * Set weight and bias for each neuron in net
    */
    void setRandNetParam(int* layerSize);

    /*
    * Set value for each neuron in net
    */
    void setLayersValues(int* x);

    /*
    * Count result error of net
    */
    double getNetError(int* y);

    /*
    * Set new error and neuron parametrs
    */
    void setNetCorrection(int* y);

    /*
    * Fisher–Yates shuffle
    */
    void shuffleTrainingSet();

    /*
    * Iteration of all training set
    */
    void correctNetOnSet();

    /*
    * Check result error for each training set
    */
    bool isNetTrained();

};

#endif
