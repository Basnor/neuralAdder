#include <iostream>
#include "Net.h"
#include <ctime>

using namespace std;

int main()
{
	setlocale(LC_ALL, "Russian");
	srand(time(0));

	Net *net = new Net;
    net->trainNet();
    net->showResultsForAllSets();
}
