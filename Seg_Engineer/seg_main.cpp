//
//  seg_main.cpp
//  Seg_Engineer
//
//  Created by Camlin Zhang on 2020/11/5.
//

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <list>

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

torch::Tensor data_preprocess(cv::Mat& input_image, torch::Device device)
{
    cv::Mat image;
    if (input_image.empty() || !input_image.data){
        std::cout << "read image fail" << std::endl;
    };

    cv::cvtColor(input_image, image, cv::COLOR_BGR2RGB);
    int height = image.rows;
    int width = image.cols;
    int depth = image.channels();
    
    std::cout << image.rows << ' ' <<  image.cols << ' ' <<  image.channels() << std::endl;
    std::cout << "read image ok!!!" << std::endl;

    // resize(256)
    cv::Size scale = cv::Size(576, 1024);
    cv::resize(image, image, scale, 0, 0, cv::INTER_LINEAR);
    std::cout << image.rows << ' ' <<  image.cols << ' ' <<  image.channels() << std::endl;

    // 转换 [unsigned int] to [float]
//    image.convertTo(image, CV_32FC3, 1.0 / 255.0); 这样写跑不通
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte).to(device);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    std::cout << tensor_image.sizes() << std::endl;

    // transforms.Normalize(mean=[0.485, 0.456, 0.406],
    //                    std=[0.229, 0.224, 0.225])
    tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
    tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
    tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
//    std::cout << tensor_image.sizes() << std::endl;
//    std::cout << tensor_image[0][0][0][0] << ' ' << tensor_image[0][0][0][1] << ' '  << tensor_image[0][0][0][2] << '\n';
    
//    cv::Mat o_Mat(cv::Size(576, 1024), CV_32F, tensor_image.data_ptr());
//    std::cout << o_Mat.rows << ' ' <<  o_Mat.cols << ' ' <<  o_Mat.channels() << std::endl;
//    cv::imwrite("debug.jpg", o_Mat);
    
    return tensor_image;
}

int main() {
    
    // 读取图片
    std::cout << "read image ..." << std::endl;
    string file_name = "/path/to/img/file";
    string file_out_name = "/path/to/img/out/file";
    cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
    
    //设置device类型//
    torch::DeviceType device_type;
    device_type = torch::kCPU;
    torch::Device device(device_type);

    //读取模型
    std::cout << "Load model ..." << std::endl;
    torch::NoGradGuard no_grad;
    torch::jit::script::Module module = torch::jit::load("/path/to/model/file");
    module.to(device);
    module.eval();

    std::cout << "Data preprocess ..." << std::endl;
//    //对图片进行处理，得到张量
    torch::Tensor img_var = data_preprocess(image, device);
    
    for(int i = 250; i <= 300; i++){
        std::cout << img_var[0][1][32][i] << ' ' << std::endl;
    }
    
    img_var = img_var.to(torch::kCPU);

    //进行预测
    std::cout << "Predict ..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();//记录时间

    torch::Tensor result_out = module.forward({ img_var }).toTensor();
    std::cout << "result ..." << result_out.sizes() << std::endl;
//    torch::Tensor tensor_image = result_encoder.squeeze(0);
//    std::cout << result[0][0][0] << std::endl;

//    std::list<torch::Tensor> result_middle;
//    result_middle.push_back(result_encoder);
//    torch::Tensor result_decoder = module_decoder.forward({ result_encoder }).toTensor();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000.0 << " sec" << std::endl;


    torch::Tensor result = result_out.sigmoid();
    result = result.squeeze();//删除一个维度，默认是squeeze(0)第1维
    result = result.mul(255).to(torch::kU8);
    result = result.to(torch::kCPU);
    cv::Mat pts_mat(cv::Size(544, 960), CV_8U, result.data_ptr());

    cv::imwrite(file_out_name, pts_mat);

    return 0;
}

