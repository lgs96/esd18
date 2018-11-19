#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>

#include "cnn.h"
#include "config.h"

using namespace std;
w_t test_image[1000][1 * 28 * 28];
double total=0;
double correct = 0;

void conv(w_t *image,													// input image
		pair<uint32_t, uint32_t> image_size,							// input image size
		uint32_t num_features,										// number of features in input = channel
		w_t *filter,													// convolution filter source
		w_t *bias,													// convolution bias source
		uint32_t num_filters,											// number of output 
		w_t *feature_map,												// output image
		pair<uint32_t, uint32_t> filter_size,							// filter size
		int32_t pad,													// number of padding
		uint32_t stride) {											// number of stride

	//define image size
	uint32_t row_img = image_size.first;
	uint32_t col_img = image_size.second;

	//define filter size
	uint32_t row_fil = filter_size.first;
	uint32_t col_fil = filter_size.second; 

	//define output image size
	uint32_t newrow = floor ((row_img + 2*pad - row_fil)/stride) + 1;
	uint32_t newcol = floor	((col_img + 2*pad - col_fil)/stride) + 1;		

	//define index
	uint32_t idx_img;
	uint32_t idx_row;
	uint32_t idx_col;

	//define 1-d array with zero padding
	//w_t * pad_img = new w_t[(row_img+pad)*(col_img+pad)];  

	//convolution calculation
	for(uint32_t a = 0; a < num_filters; a++){
		for (uint32_t c = 0; c < newrow; c++) {
			for (uint32_t d = 0; d < newcol; d++) {
				w_t val = 0;
				for (uint32_t b = 0; b < num_features; b++) {
					idx_img = b * row_img*col_img;
					idx_row = c * col_img * stride;
					idx_col = d * stride;
					for (uint32_t e = 0; e < row_fil; e++) {
						for (uint32_t f = 0; f < col_fil; f++) {

							val += image[idx_img + idx_row + idx_col + e * col_img + f] * filter[a*row_fil*col_fil*num_features+b*row_fil*col_fil + e * col_fil + f];

							//							cout<<"val is "<<val<<" image val is "<<image[idx_fil+idx_row+idx_col+e*col_img+f]<<" filter val is "<<filter[b*row_fil*col_fil+e*col_fil+f]<<endl;	
						}
					}
				}
				idx_img = a * newrow*newcol;
				idx_row = c * newcol;
				idx_col = d;
				val += bias[a];
				feature_map[idx_img + idx_row + idx_col] = val;
			}
		}
	}
	//delete [] pad_img;
}

void max_pool(w_t *image,												// input image
		pair<uint32_t, uint32_t> image_size,		// input image size
		uint32_t channel,							// number of features in input image = channel
		pair<uint32_t, uint32_t> max_pool_size,		// pooling size
		uint32_t stride,							// stride
		w_t *max_pool) {							// output image

	//define image size
	uint32_t row_img = image_size.first;
	uint32_t col_img = image_size.second;

	//define filter size
	uint32_t row_fil = max_pool_size.first;
	uint32_t col_fil = max_pool_size.second; 

	//define output image size
	uint32_t newrow = floor ((row_img - row_fil)/stride) + 1;
	uint32_t newcol = floor	((col_img - col_fil)/stride) + 1;		

	//define index
	uint32_t idx_img;
	uint32_t idx_row;
	uint32_t idx_col;
	uint32_t new_idx;

	//define 1-d array with zero padding
	//w_t * pad_img = new w_t[(row_img+pad)*(col_img+pad)];  
	//cout << newrow << " " << newcol << endl;
	//convolution calculation
	for(uint32_t a = 0; a < channel; a++){
		for (uint32_t c = 0; c < newrow; c++){	
			for(uint32_t d = 0 ; d < newcol; d++){
				w_t val = -9999;
				idx_img = a * row_img*col_img;
				idx_row = c * col_img*stride;
				idx_col = d * stride;
				for(uint32_t e = 0 ; e < row_fil; e++){
					for(uint32_t f = 0; f < col_fil; f++){
		
						if (val < image[idx_img + idx_row + idx_col + f + e * col_img]) {
							val = image[idx_img + idx_row + idx_col + f + e * col_img];
						}
					//	cout << "val is " << val << " image val is " << image[idx_img + idx_row + idx_col + e * col_img + f] <<" index:"<< idx_img + idx_row + idx_col + e * col_img + f<<endl;
					
					} 
				}
				idx_img = a*newrow*newcol;
				idx_row = c*newcol;
				idx_col = d;
				new_idx = idx_img + idx_row + idx_col;

				max_pool[new_idx] = val; 				 
			}	
		} 				 
	}
}


void ReLu(w_t *image,													// input image
		pair<uint32_t, uint32_t> image_size,					// input image size
		uint32_t num_output,									// number of output feature			
		w_t *output) {											// output

	for(uint32_t i = 0; i <num_output*image_size.first*image_size.second; i++){
		if(image[i]<0){
	//		cout<<i<<"image was "<<image[i]<<endl;
			output[i] = 0;
	//		cout<<i<<"image is "<<image[i]<<endl;
		}
		else{
			output[i]=image[i];
		}
	}
}

void TanH(w_t *image, 													// input image
		pair<uint32_t, uint32_t> image_size,					// input image size
		uint32_t num_output,									// number of output feature
		w_t *output){											// output
	//FIXME
}

void ip(w_t *input, pair<uint32_t, uint32_t> input_size,				// input image
		uint32_t num_features,									// number of 1 input's features
		w_t *weight,											// weights
		w_t *bias,												// bias
		uint32_t num_output,									// number of output neurons
		w_t *output){								//output
	//cout << "i want"<<num_features<<endl;
	for(uint32_t i = 0; i < num_output; i++){
		output[i] = 0;
		uint32_t tensorsize = num_features*input_size.first*input_size.second;
		for(uint32_t j = 0; j < tensorsize; j++){
			output[i] += input[j]*weight[i*tensorsize + j];
		}
		output[i] += bias[i];			
	}									
}

void accuracy(uint32_t iter,											// number of iterations
		uint32_t *label,							// label for test data
		w_t *output){			
	// expected outputs
	w_t predict = -1000;
	w_t predict_num = -1;
	
	for (uint32_t i = 0; i < 10; i++) {
		if (predict < output[i]) {
			predict = output[i];
			predict_num = i;
		}
	}
	/*
	cout << "value is..." << output[predict_num] << endl;
		cout << "predict is..." << predict_num << endl;
		cout << "answer is..." << label[iter] << endl;
		*/
	total++;
	if (predict_num == label[iter]) {
		correct=correct+1;
	}

	if (total == 1000) {
		cout << correct*100/total <<"% accuracy"<<endl;
	}
	
	
}

