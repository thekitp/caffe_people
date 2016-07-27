//
// Created by aurora on 16-7-27.
//

#include <iostream>
#include "caffe/caffe.hpp"
#include "caffe/data_transformer.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;

#define PRINT_SHAPE1(x) \
    std::cout << (x).num() << "\t" << (x).channels() << "\t" << (x).height() << "\t" << (x).width() << "\n";
#define PRINT_SHAPE2(x) \
    std::cout << (x)->num() << "\t" << (x)->channels() << "\t" << (x)->height() << "\t" << (x)->width() << "\n";
#define PRINT_DATA(x) \
    std::cout << (x)[0] << "\t" << (x)[1] << "\n";

cv::Size input_geometry_;
TransformationParameter transform_param_;
shared_ptr<DataTransformer<float> > data_transformer_;
Blob<float> transformed_data_;


void show_blob(Blob<float>* data){
    for(int i=0; i<data->num(); i++){
        for(int j=0; j<1; j++){
            for(int k=0; k<data->height(); k++){
                for(int d=0; d<data->width(); d++){
                    std::cout<<data->data_at(i, j, k, d)<<",";
                }
            }
        }
    }
}

void readImageToBlob(const string &img_path, int idx, bool is_color, Blob<float>* input_blob_) {
// read the image and resize to the target size
    cv::Mat img;
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat cv_img_origin = cv::imread(img_path, cv_read_flag);
    if (!cv_img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << img_path;
        return ;
    }
    if (input_geometry_.height > 0 && input_geometry_.width > 0) {
        cv::resize(cv_img_origin, img, input_geometry_);
    } else {
        img = cv_img_origin;
    }

    transformed_data_.Reshape(input_blob_->shape());
    // transform the image to a blob using DataTransformer
    // create a DataTransformer using default TransformationParameter (no transformation)
    data_transformer_.reset(new DataTransformer<float>(transform_param_, TEST));
    data_transformer_->InitRand();
    // set the output of DataTransformer to the idx image of the input blob
    int offset = input_blob_->offset(idx);
    transformed_data_.set_cpu_data(input_blob_->mutable_cpu_data() + offset);
//    transformed_data_.set_gpu_data(input_blob_->mutable_cpu_data() + offset);
    // transform the input image
    data_transformer_->Transform(img, &(transformed_data_));
    show_blob(&transformed_data_);
}


Blob<float>* processImage(shared_ptr<Net<float> > net_, Blob<float>* input_blob_, const string &img_path, bool is_color) {
// reshape the net for the input
    input_blob_ = net_->input_blobs()[0];
    input_blob_->Reshape(1, 3, input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    readImageToBlob(img_path, 0, is_color, input_blob_);
    vector<Blob<float>*> out = net_->Forward();

    return out[0];
}

int main(){
    string net_proto = "/home/aurora/hdd/workspace/ClionProjects/Caffe_People/networks/aur_deploy.prototxt";
    string image_url = "/home/aurora/hdd/workspace/ClionProjects/Caffe_People/networks/cat.jpg";

    Phase phase = TEST;
//    Caffe::set_mode(Caffe::CPU);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

    boost::shared_ptr<Net<float> > net(new caffe::Net<float>(net_proto, phase));
    const std::vector<boost::shared_ptr<Layer<float>  >> layers = net->layers();
    std::vector<boost::shared_ptr<Blob<float>  >> net_blobs = net->blobs();
    std::vector<string> layer_names = net->layer_names();
    std::vector<string> blob_names = net->blob_names();
    boost::shared_ptr<Layer<float> > layer;
    boost::shared_ptr<Blob<float> > blob;

    // show input blob size
    Blob<float>* input_blobs = net->input_blobs()[0];
    std::cout << "\nInput blob size:\n";
    PRINT_SHAPE2(input_blobs);
    input_geometry_ = cv::Size(input_blobs->width(), input_blobs->height());
    Blob<float>* result = processImage(net, input_blobs, image_url, true);

    std::cout << "result count is ====" << result->count()<<std::endl;
    show_blob(result);
//    int count = net->input_blobs()[0]->count();
//    std::cout<<"input count is ==="<<count<<std::endl;
//    const float* output = net->output_blobs()[0]->cpu_data();
//    std::cout<<"output count is ====="<<net->output_blobs()[0]->count();

}