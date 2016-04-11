//
// Created by aurora on 16-4-11.
//

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
//using namespace cv;

//int main(){
//    cvNamedWindow("video");
//    CvCapture* capture= cvCreateCameraCapture(-1);
//    IplImage* frame;
//    while(1)
//    {
//        frame=cvQueryFrame(capture);
//        if(!frame) break;
//        cvShowImage("win",frame);
//        char c=cvWaitKey(10);
//        if(c==27) break;
//    }
//    cvReleaseCapture(&capture);
//    cvDestroyWindow("win");
//    return 0;
//}

#include <cstdio>
#include <iostream>
#include <ctime>
#include "HCNetSDK.h"


using namespace cv;
using namespace std;


//typedef HWND (WINAPI *PROCGETCONSOLEWINDOW)();
//PROCGETCONSOLEWINDOW GetConsoleWindow;

int main(int argc, char * argv[])
{
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
    lUserID = NET_DVR_Login_V30("192.168.3.38", 8000, "admin", "tdabcd1234", &struDeviceInfo);
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
        cout<<"Set Capture Picture Mode error!"<<endl;
        cout<<"The error code is "<<NET_DVR_GetLastError()<<endl;
    }

    //bool capture = NET_DVR_CaptureJPEGPicture(lUserID,1,JpegPara,"1111");
    vector<char> data(200*1024);
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

        Mat img = imdecode(Mat(data),1);
        imshow("camera",img);
        waitKey(1);

    }

    //FILE * fp = fopen("3.jpg","wb");
    //fwrite(Jpeg,1,123*1024,fp);
    //fclose(fp);

    return 0;
}
