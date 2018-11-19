#include "config.h"

using namespace std;

float feature_map1[26*26*5];
float ReLU_map1[26*26*5];
float feature_map2[24*24*5];
float ReLU_map2[24*24*5];
float max_pool1[12*12*5];
float feature_map3[10*10*5];
float ReLU_map3[10*10*5];
float feature_map4[8*8*5];
float ReLU_map4[8*8*5];
float max_pool2[4*4*5];
float ip1[10];
float tanh1[40];
float ip2[10];
