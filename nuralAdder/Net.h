#ifndef NET_H
#define NET_H

#include <iostream>
#include "Layer.h"
#include <math.h>

#define X_NUM 3
#define Y_NUM 3
#define SAMPLE_SIZE 7

class Net
{
public:
	Net();

private:
	Layer* m_layer;
	Layer* m_inputLayer;

	void setRandNet(int* layerSize, int layerNum);
	void setLayersValue(int* x, int layerNum);
	double getNetError(int* y, int layerNum);
	void setNetCorrection(int* y, int layerNum);
};

#endif
