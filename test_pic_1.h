//
// Created by aurora on 16-4-10.
//

#ifndef CAFFE_MANUAL2_TEST_PIC_1_H
#define CAFFE_MANUAL2_TEST_PIC_1_H

#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "boost/algorithm/string.hpp"

//caffe
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/layers/memory_data_layer.hpp"

using boost::shared_ptr;
using std::string;
using namespace caffe;

#define PRINT_SHAPE1(x) \
    LOG(ERROR) << (x).num() << "\t" << (x).channels() << "\t" << (x).height() << "\t" << (x).width() << "\n";
#define PRINT_SHAPE2(x) \
    LOG(ERROR) << (x)->num() << "\t" << (x)->channels() << "\t" << (x)->height() << "\t" << (x)->width() << "\n";
#define PRINT_DATA(x) \
    LOG(ERROR) << (x)[0] << "\t" << (x)[1] << "\t" <<(x)[2]<< "\n";

class PID_1 {
public:
    PID_1();
    PID_1(string n_proto, string n_model, string n_mean_file, string mode, uint device_id, Phase phase, string _blob_names);
    ~PID_1();
    void caffe_forward(boost::shared_ptr<Net<float>> &net, float *data_ptr);
    unsigned  int get_blob_index(boost::shared_ptr<Net<float>> &net, string query_blob_name);
    unsigned int get_layer_index(boost::shared_ptr< Net<float> > & net, char *query_layer_name);
    void get_image_mean();
    void init_net();
    const std::vector<boost::shared_ptr<Layer<float>>>& get_layers() const;
    void caculat_result(cv::Mat& img, int label, string filepath);

    void GetAllFiles(string path, vector<string>& files);
//    void GetAllFiles(string path);
    void init_image(string filepath);
private:
    string _proto;
    string _model;
    string _mean_file;
    string _blob_names;

    string _mode;
    uint _device_id;

    //image mean
    Blob<float> image_mean;
    BlobProto blob_proto;
    const float *mean_ptr;

    Phase _phase;
    shared_ptr<Net<float>> net;
    std::vector<boost::shared_ptr<Layer<float>>> layers;
    std::vector<boost::shared_ptr<Blob<float>>> net_blobs;
    std::vector<string> layer_names;
    std::vector<string> blob_names;
    boost::shared_ptr<Layer<float> > layer;
    boost::shared_ptr<Blob<float> > blob;

    std::vector<cv::Mat> _images;
};


#endif //CAFFE_MANUAL2_TEST_PIC_1_H
