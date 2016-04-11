#include <string>
#include <vector>
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include <string>
using boost::shared_ptr;
using std::string;
using namespace caffe;

#define MAX_FEAT_NUM 16

//! Note: data_ptr指向已经处理好（去均值的，符合网络输入图像的长宽和Batch Size）的数据
void caffe_forward(boost::shared_ptr<Net<float>> &net, float *data_ptr){
    Blob<float>* input_blobs = net->input_blobs()[0];
    switch (Caffe::mode()){
        case Caffe::CPU:
            memcpy(input_blobs->mutable_cpu_data(), data_ptr, sizeof(float)*input_blobs->count());
            break;
        case Caffe::GPU:
            cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr, sizeof(float)*input_blobs->count(), cudaMemcpyHostToDevice);
            break;
        default:
            LOG(FATAL)<<"Unknow Caffe Mode. ";
    }
    net->ForwardPrefilled();
}

//! Note: Net的Blob是指，每个层的输出数据，即Feature Maps
// char *query_blob_name = "conv1";
unsigned  int get_blob_index(boost::shared_ptr<Net<float>> &net, string query_blob_name){
    string str_query(query_blob_name);
    const vector<string>  &blob_names = net->blob_names();
    for(unsigned int i=0; i!=blob_names.size(); ++i){
        if(str_query == blob_names[i]){
            return i;
        }
    }
    LOG(FATAL)<<"Unknown blob name: "<<str_query;
}


//! Note: Layer包括神经网络所有层，比如，CaffeNet共有23层
// char *query_layer_name = "conv1";
unsigned int get_layer_index(boost::shared_ptr< Net<float> > & net, char *query_layer_name)
{
    std::string str_query(query_layer_name);
    vector< string > const & layer_names = net->layer_names();
    for( unsigned int i = 0; i != layer_names.size(); ++i )
    {
        if( str_query == layer_names[i] )
        {
            return i;
        }
    }
    LOG(FATAL) << "Unknown layer name: " << str_query;
}

int main(int argc, char** argv)
{
    string proto = "/home/aurora/hdd/software/Caffe/caffe-master/examples/people_identity/deploy.prototxt";
    string model = "/home/aurora/hdd/software/Caffe/caffe-master/examples/people_identity/caffenet_train_iter_150000.caffemodel";
    string mean_file = "/home/aurora/hdd/software/Caffe/caffe-master/examples/people_identity/lmdb/people_identity_mean.binaryproto";
    Phase phase = TEST;
    Caffe::set_mode(Caffe::GPU);
    boost::shared_ptr<Net<float>> net(new caffe::Net<float>(proto, phase));
    net->CopyTrainedLayersFrom(model);

    //read blob info  feature map
//    string query_blob_name = "conv1";
//    unsigned int blob_id = get_blob_index(net, query_blob_name);
//    boost::shared_ptr<Blob<float>> blob = net->blobs()[blob_id];
//    unsigned int num_data = blob->count();
//    std::cout<<num_data;
//    const float * blob_ptr = (const float *)blob->cpu_data();

    //read layer info
    //! Note: 不同于Net的Blob是Feature Maps，Layer的Blob是指Conv和FC等层的Weight和Bias
//    char *query_layer_name = "conv1";
//    const float *weight_ptr, *bias_ptr;
//    unsigned int layer_id = get_layer_index(net, query_layer_name);
//    boost::shared_ptr<Layer<float> > layer = net->layers()[layer_id];
//    std::vector<boost::shared_ptr<Blob<float>  >> blobs = layer->blobs();
//    if (blobs.size() > 0)
//    {
//        weight_ptr = (const float *) blobs[0]->cpu_data();
//        std::cout<< sizeof(weight_ptr)/sizeof(weight_ptr[0]);
//        bias_ptr = (const float *) blobs[1]->cpu_data();
//    }
//! Note: 训练模式下，读取指定Layer的梯度数据，与此相似，唯一的区别是将cpu_data改为cpu_diff

    char *query_layer_name = "conv1";
    const float* data_ptr;          /* 指向待写入数据的指针， 源数据指针*/
    float* weight_ptr = NULL;       /* 指向网络中某层权重的指针，目标数据指针*/
    unsigned int data_size;         /* 待写入的数据量 */
    char *layer_name = "conv1";     /* 需要修改的Layer名字 */

    unsigned int layer_id = get_layer_index(net, query_layer_name);
    boost::shared_ptr<Blob<float> > blob = net->layers()[layer_id]->blobs()[0];

    CHECK(data_size == blob->count());
    switch (Caffe::mode())
    {
        case Caffe::CPU:
            weight_ptr = blob->mutable_cpu_data();
            break;
        case Caffe::GPU:
            weight_ptr = blob->mutable_gpu_data();
            break;
        default:
            LOG(FATAL) << "Unknown Caffe mode";
    }
    caffe_copy(blob->count(), data_ptr, weight_ptr);

    // save new model
    char* weights_file = "bvlc_reference_caffenet_new.caffemodel";
    NetParameter net_param;
    net->ToProto(&net_param, false);
    WriteProtoToBinaryFile(net_param, weights_file);
//! Note: 训练模式下，手动修改指定Layer的梯度数据，与此相似
// mutable_cpu_data改为mutable_cpu_diff，mutable_gpu_data改为mutable_gpu_diff

//    NetParameter param;
//    ReadNetParamsFromBinaryFileOrDie(model, &param);
//    int num_layers = param.layer_size();
//    for(int i=0; i<num_layers; i++){
//        LOG(ERROR)<<"Layer "<<i<<" : "<<param.layer(i).name()<<"\t"<<param.layer(i).type();
//        if(param.layer(i).type()=="Convolution")
//        {
//            ConvolutionParameter conv_param = param.layer(i).convolution_param();
//            LOG(ERROR) << "\t\tkernel size: " << conv_param.kernel_w()<<" pad: "<<conv_param.pad_size()<<" stride: "<<conv_param.stride_size();
//        }
//    }
//
//    Blob<float> image_mean;
//    BlobProto blob_proto;
//    const float *mean_ptr;
//    unsigned  int num_pixel;
//    bool succed = ReadProtoFromBinaryFile(mean_file, &blob_proto);
//    if(succed){
//        image_mean.FromProto(blob_proto);
//        num_pixel = image_mean.count();
//        mean_ptr = (const float*)image_mean.cpu_data();
//    }
    return 0;
}
