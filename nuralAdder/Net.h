#ifndef NET_H
#define NET_H

#include <iostream>
#include "Layer.h"
#include <math.h>

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
    int m_layersNum;

	Layer* m_layer;
	Layer* m_inputLayer;

    /*
     * Set weight and bias for each neuron in net
     */
    void setRandNetParam(int* layerSize, int layerNum);
    void setLayersValues(int* x, int layerNum);
	double getNetError(int* y, int layerNum);
	void setNetCorrection(int* y, int layerNum);

    /*
     * Fisher–Yates shuffle
     */
    void shuffleTrainingSet();

    void correctNetOnSet();
    bool isNetTrained();

};

#endif
