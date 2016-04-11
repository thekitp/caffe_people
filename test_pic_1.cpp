//
// Created by aurora on 16-4-10.
//

#include "test_pic_1.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include "HCNetSDK.h"


PID_1::PID_1() {

}

PID_1::PID_1(string n_proto, string n_model, string n_mean_file, string mode, uint device_id, Phase phase, string blob_names):
        _proto(n_proto), _model(n_model), _mean_file(n_mean_file), _mode(mode), _device_id(device_id), _phase(phase), _blob_names(blob_names){

}

PID_1::~PID_1() {

}
//! Note: data_ptr指向已经处理好（去均值的，符合网络输入图像的长宽和Batch Size）的数据
void PID_1::caffe_forward(boost::shared_ptr<Net<float>> &net, float *data_ptr) {
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
unsigned int PID_1::get_blob_index(boost::shared_ptr<Net<float>> &net, string query_blob_name){
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
unsigned int PID_1::get_layer_index(boost::shared_ptr< Net<float> > & net, char *query_layer_name){
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

void PID_1::get_image_mean() {
    bool succeed = ReadProtoFromBinaryFile(_mean_file, &blob_proto);
    if (succeed)
    {
        std::cout << "read image mean succeeded" << std::endl;
        image_mean.FromProto(blob_proto);
        mean_ptr = (const float *) image_mean.cpu_data();
        int num_pixel = image_mean.count();
        std::cout << num_pixel << "\n";
        PRINT_SHAPE1(image_mean);
        PRINT_DATA(mean_ptr);
    }
    else
    {
        LOG(FATAL) << "read image mean failed";
    }
}

void PID_1::init_net() {
    if(_mode=="gpu"){
        Caffe::set_mode(Caffe::GPU);
    }else{
        Caffe::set_mode(Caffe::CPU);
    }
    net.reset(new caffe::Net<float>(_proto, _phase));
    net->CopyTrainedLayersFrom(_model);
    layers = net->layers();
    net_blobs = net->blobs();
    layer_names = net->layer_names();
    // Check that requested blobs exist
    boost::split(blob_names, _blob_names, boost::is_any_of(","));
    for (size_t i = 0; i < blob_names.size(); i++) {
        if (!net->has_blob(blob_names[i]))
        {
            LOG(FATAL) << "Unknown feature blob name " << blob_names[i] << " in the network " << _proto;
        }
    }
}

const std::vector<boost::shared_ptr<Layer<float>>>& PID_1::get_layers() const {
    return layers;
}

//获取所有的文件名
void PID_1::GetAllFiles(string path, vector<string>& files)
{
    DIR *dp;
    struct dirent *dirp;
    int n=0;
    const char *filePath = path.c_str();
    if((dp=opendir(filePath))==NULL)
        printf("can't open %s",filePath);
    while (((dirp=readdir(dp))!=NULL))
    {
        char *p = dirp->d_name;
        if(strcmp(p,".") != 0  &&  strcmp(p,"..") != 0){
            files.push_back(path+p);
        }
//        printf("%s\n ",path+filename);
    }
    closedir(dp);
}


void PID_1::init_image(string filepath) {

}

void PID_1::caculat_result(cv::Mat& img, int label, string filepath="video") {
    vector<cv::Mat> mat_vec;
    vector<int> label_vec;
    mat_vec.push_back(img);
    label_vec.push_back(label);
    shared_ptr<MemoryDataLayer<float>> memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(layers[0]);
//    LOG(ERROR)<< memory_data_layer->batch_size();
//    LOG(ERROR)<<"image rows\t"<< img.rows << " image heights\t"<<img.cols;
    memory_data_layer->AddMatVector(mat_vec, label_vec);

    size_t num_features = blob_names.size();
    std::vector<Blob<float>*> results;
    results = net->Forward();
    for(int i=0; i<num_features; i++){
        const shared_ptr<Blob<float> > feature_blob = net->blob_by_name(blob_names[i]);
        int batch_size = feature_blob->num();
        for(int j=0; j<batch_size; j++){
//            LOG(ERROR)<<sizeof(feature_blob->cpu_data())/sizeof(feature_blob->cpu_data()[0]);
//            LOG(ERROR)<<feature_blob->cpu_data()[0];
//            LOG(ERROR)<<feature_blob->cpu_data()[1];
            if(feature_blob->cpu_data()[0]>feature_blob->cpu_data()[1]){
                LOG(ERROR)<<"000" <<" image "<< filepath << " contains no people";
            }else{
                LOG(ERROR)<<"111" << "image "<< filepath << " contains people";
            }
        }
    }
}

int main(){
    string proto = "/home/aurora/hdd/software/Caffe/caffe-master/examples/people_identity/deploy2.prototxt";
    string model = "/home/aurora/hdd/software/Caffe/caffe-master/examples/people_identity/caffenet_train_iter_150000.caffemodel";
    string mean_file = "/home/aurora/hdd/software/Caffe/caffe-master/examples/people_identity/lmdb/people_identity_mean.binaryproto";

    //image information
    string img_location = "/home/aurora/hdd/software/Caffe/caffe-master/data/peoples_identity/valdatas256/nopic00738.jpg";
    string image_path = "/home/aurora/hdd/software/Caffe/caffe-master/data/peoples_identity/valdatas256/";

    cv::Mat img = cv::imread(img_location);
    int label = 0;
//
    PID_1 pid(proto, model, mean_file, "cpu", 0, TEST, "prob");
    vector<string> filenames;
    pid.get_image_mean();
    pid.init_net();
    // Get pointer to data layer to set the input
//    pid.GetAllFiles(image_path, filenames);
//    for(string file:filenames){
//        img = cv::imread(file);
//        pid.caculat_result(img, label, file);
//    }
//    pid.caculat_result(img, label, img_location);


    //video process
    //---------------------------------------
    // 初始化
    NET_DVR_Init();
    //设置连接时间与重连时间
    NET_DVR_SetConnectTime(2000, 1);
    NET_DVR_SetReconnect(10000, true);

    //---------------------------------------
    //获取控制台窗口句柄
    //HMODULE hKernel32 = GetModuleHandle((LPCWSTR)"kernel32");
    //GetConsoleWindow = (PROCGETCONSOLEWINDOW)GetProcAddress(hKernel32,"GetConsoleWindow");


    //---------------------------------------
    // 注册设备
    LONG lUserID;
    NET_DVR_DEVICEINFO_V30 struDeviceInfo;
    lUserID = NET_DVR_Login_V30("192.168.3.41", 8000, "admin", "tdabcd1234", &struDeviceInfo);
    if (lUserID < 0)
    {
        printf("Login error, %d\n", NET_DVR_GetLastError());
        NET_DVR_Cleanup();
        return -1;
    }


    //---------------------------------------
    //cvNamedWindow("camera",CV_WINDOW_AUTOSIZE);
    IplImage* frame;
    //定义JPEG图像质量
    LPNET_DVR_JPEGPARA JpegPara = new NET_DVR_JPEGPARA;
    JpegPara->wPicQuality = 0;
    JpegPara->wPicSize = 9;

    char * Jpeg = new char[200*1024];
    DWORD len = 200*1024;
    LPDWORD Ret = 0;

    if(!NET_DVR_SetCapturePictureMode(BMP_MODE))
    {
        std::cout<<"Set Capture Picture Mode error!"<<std::endl;
        std::cout<<"The error code is "<<NET_DVR_GetLastError()<<std::endl;
    }

    //bool capture = NET_DVR_CaptureJPEGPicture(lUserID,1,JpegPara,"1111");
    vector<char>data(200*1024);
    cv::Size dsize = cv::Size(256, 256);
    while(1)
    {
        bool capture = NET_DVR_CaptureJPEGPicture_NEW(lUserID,1,JpegPara,Jpeg,len,Ret);
        if(!capture)
        {
            printf("Error: NET_DVR_CaptureJPEGPicture_NEW = %d", NET_DVR_GetLastError());
            return -1;
        }

        for(int i=0;i<200*1024;i++)
            data[i] = Jpeg[i];

        cv::Mat img = cv::imdecode(cv::Mat(data),1);
        cv::Mat img2 = cv::Mat(dsize, img.type());
        cv::resize(img, img2, dsize);
        pid.caculat_result(img2, label, "video");
        imshow("camera",img);
        cv::waitKey(1);

    }


    return 0;
}