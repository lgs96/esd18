#include <iostream>

#include "cnn.h"
#include "config.h"
#include "math.h"
#include "./data/weights_conv1.h"
#include "./data/weights_conv2.h"
#include "./data/weights_conv3.h"
#include "./data/weights_conv4.h"
#include "./data/weights_ip.h"
#include "./data/bias_conv1.h"
#include "./data/bias_conv2.h"
#include "./data/bias_conv3.h"
#include "./data/bias_conv4.h"
#include "./data/bias_ip.h"
#include "./data/test_set.h"
#include "./data/label.h"

#define TS 780*1000

using namespace std;

int main() { 	
	cout<<"##########Welcome to the first project fo ESD18"<<endl;
	w_t mean = 0;
	w_t square_mean = 0;
	w_t std;

	//Data pre-processing (Normalization)
	for(uint32_t i = 0; i < TS; i++){
		mean = mean + ts[i];
		square_mean = square_mean + (ts[i])*(ts[i]);		
	}

	mean = mean/(TS);
	square_mean = square_mean/(TS);
	std = sqrtf(square_mean - mean*mean);

	cout << mean << "\t" << std << endl;

	for(uint32_t i = 0; i <TS; i++){
		ts[i] = (ts[i]-mean)/std;
//		cout<<ts[i]<<endl;
	}

	//Neural network of HW#1 
	//layer: conv-relu-conv-relu-pool
	//network: layer1-layer2-fc
	for (uint32_t iter =0; iter <1000; ++iter){
		for (uint32_t i =0 ; i < 784; ++i){
			test_image[iter][i] = ts[784 * iter + i];		
		}
		/*
			w_t test[32] = {1,3,5,7,9,1,3,5,7,9,1,3,5,7,9,1,3,5,3,4,5,6,7,2,4,6,7,2,3,6,7,8};
				w_t weight[16] = {1, 2, 3, 4,4,3,2,1,1, 2, 3, 4,4,3,2,1 };
				w_t bias[2] = {0, 0}; 
				w_t out[16];		

				
				conv(test,make_pair(3,3),2,
				weight,
				bias,2,
				out,
				make_pair(2,2),0,1);
			
			for (uint32_t i = 0; i < 8; i++) {
			cout <<"out is "<< out[i] << endl;
			}
			
				max_pool(test, make_pair(4, 4), 2,
					make_pair(2, 2), 2, out);
					
				for (uint32_t i = 0; i < 16; i++) {
					cout << "out is " << out[i] << endl;
				}
*/
	

		conv(test_image[iter], make_pair(28, 28), 1,
				_weights_conv1,
				_bias_conv1, 5,
				feature_map1,
				make_pair(3, 3), 0, 1);

		conv(feature_map1, make_pair(26, 26), 5,
			_weights_conv2,
			_bias_conv2, 5,
			feature_map2,
			make_pair(3, 3), 0, 1);

		max_pool(feature_map2, make_pair(24, 24), 5,
			make_pair(2, 2), 2, max_pool1);
/*
		for (uint32_t i = 0; i < 12; i++) {
			for (uint32_t j = 0; j < 12; j++) {
				max
			}
		}
*/
		ip(max_pool1,
			make_pair(12, 12), 5,
			_weights_ip,
			_bias_ip,
			10, ip1);
	/*
		 for(uint32_t i = 0; i <26*26*5; i++){
		  	 cout<<i<<"image is "<<feature_map1[i]<<endl;
		 }*/
		 /*
		 ip(feature_map1,
			 make_pair(26, 26), 5,
			 _weights_ip,
			 _bias_ip,
			 10, ip1);*/
	/*
		ReLu(feature_map1,
				make_pair(26,26),
				5,
				ReLU_map1);
		for (uint32_t i = 0; i <5 * 26 * 26 * 5; i++) {
			cout << i << "image is " << ReLU_map1[i] << endl;
		}
		conv(ReLU_map1, make_pair(26, 26), 5,
				_weights_conv2,
				_bias_conv2, 5,
				feature_map2,
				make_pair(3, 3), 0, 1);
	
		 for(uint32_t i = 0; i <5*26*26*5; i++){
		  	 cout<<i<<"image is "<<feature_map2[i]<<endl;
		 }
		*/
	/*	ReLu(feature_map2,
				make_pair(24,24),
				5,
				ReLU_map2);
		
		for (uint32_t i = 0; i <5 * 24 * 24; i++) {
			cout << i << "image is " << ReLU_map2[i] << endl;
		}
		
		max_pool(ReLU_map2, make_pair(24, 24), 5,
				make_pair(2, 2), 2, max_pool1);

		conv(max_pool1, make_pair(12, 12), 5,
				_weights_conv3,
				_bias_conv3, 5,
				feature_map3,
				make_pair(3, 3), 0, 1);

		ReLu(feature_map3,
				make_pair(10,10),
				5,
				ReLU_map3);

		conv(ReLU_map3, make_pair(10, 10), 1,
				_weights_conv4,
				_bias_conv4, 5,
				feature_map4,
				make_pair(3, 3), 0, 1);

		ReLu(feature_map4,
				make_pair(8,8),
				5,
				ReLU_map4);

		max_pool(ReLU_map4, make_pair(8, 8), 5,
				make_pair(2, 2), 2, max_pool2);
		
		ip(max_pool2,
				make_pair(4, 4), 5,
				_weights_ip,
				_bias_ip,
				10, ip1);*/

		accuracy(iter, ls, ip1);

	}
	return 0;
}
