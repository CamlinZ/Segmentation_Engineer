# 语义分割算法(pytorch版本)C++部署全流程解析

本文以https://github.com/CSAILVision/semantic-segmentation-pytorch作为示例语义分割算法，采用libtorch来进行C++部署

整个部署分为以下几个步骤进行展开：

- 示例语义分割算法简介
- 原始模型预处理opencv化

- 模型网络结构重构
- 模型转化
- C++数据预处理、前向传播及后处理代码开发
- opencv和libtorch联合编译

## 示例语义分割算法简介

https://github.com/CSAILVision/semantic-segmentation-pytorch整体工程代码经过简化整理后如下结构所示：

```
Seg_Engeneer
|-- ckpt		训练模型输出文件
|-- config		yaml自定义模型配置文件
|   `-- my-resnet101-upernet.yaml
|-- mit_semseg		网络结构代码库
|   |-- config		yaml中默认配置文件
|   |   |-- defaults.py
|   |   |-- __init__.py
|   |-- dataset.py		数据预处理文件
|   |-- __init__.py
|   |-- lib		作者自定义的相关库文件
|   |   |-- __init__.py
|   |   |-- nn
|   |   `-- utils
|   |-- models		网络结构文件
|   |   |-- hrnet.py
|   |   |-- __init__.py
|   |   |-- mobilenet.py
|   |   |-- model_build.py		第二步中将models.py中encoder和decoder合并网络结构后的文件
|   |   |-- models.py
|   |   |-- resnet.py
|   |   |-- resnext.py
|   |   `-- utils.py
|   |-- opencv_transforms		第一步进行的原始模型预处理opencv化的依赖文件
|   |   |-- functional.py
|   |   |-- __init__.py
|   |   `-- transforms.py
|   `-- utils.py
|-- pretrained		预训练模型
|   `-- resnet101-imagenet.pth
|-- train.py		训练代码
|-- test_img.py		单张图像测试代码
```



## 原始模型预处理opencv化

python版本的pytorch中针对图像任务通常采用torchvision来进行图像的预处理，而torchvision中的预处理函数是调用PIL库进行开发。由于python下的PIL库没有对应的C++版本，而opencv是同时有Python和C++版本的接口函数，所以这里为了保证模型的精度，需要将模型的预处理函数用opencv进行重构。

这里主要参考：https://github.com/jbohnslav/opencv_transforms

这里将torchvision中对应的PIL数据预处理操作全部转换为opencv，从接口函数的列表中可以看到是可以包含几乎所有的数据预处理操作的，同时接口的参数和torchvision基本一致。

```python
__all__ = [
    "Compose", "ToTensor", "Normalize", "Resize", "Scale",
    "CenterCrop", "Pad", "Lambda", "RandomApply", "RandomChoice",
    "RandomOrder", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
    "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop",
    "LinearTransformation", "ColorJitter", "RandomRotation", "RandomAffine",
    "Grayscale", "RandomGrayscale"
]
```

本文示范的语义分割算法中，需要修改的代码主要涉及dataset.py文件，该文件中主要涉及到的是训练，验证和测试集的数据预处理操作，这里以训练数据的预处理操作进行示例：

这里需要将上述的opencv操作代码opencv_transforms按照第一步中展示的工程结构放入到工程中，将头文件改为

```
from torchvision import transforms ---> from .opencv_transforms import transforms
```

然后需要将resize函数修改为opencv的resize：

```
def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = cv2.INTER_NEAREST
    elif interp == 'bilinear':
        resample = cv2.INTER_LINEAR
    elif interp == 'bicubic':
        resample = cv2.INTER_CUBIC
    else:
        raise Exception('resample method undefined!')
    img_resized = cv2.resize(im, size, interpolation=resample)
    return img_resized
```

最后需要对TrainDataset类中涉及到PIL的操作进行修改，这里主要涉及到\__getitem__函数，经过修改后为：

```python
    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            ############## 将PIL操作转换为opencv操作 ##############
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segm = cv2.imread(segm_path, 0)

            assert(len(segm.shape) == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = cv2.flip(img, 1)
                segm = cv2.flip(segm, 1)

            # pdb.set_trace()
            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.shape[1], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.shape[0], self.segm_downsampling_rate)

            segm = imresize(
                segm,
                (segm_rounded_width // self.segm_downsampling_rate, \
                 segm_rounded_height // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output
```



## 模型网络结构重构

本次示例的语义分割网络中，作者为了方便选用不同的encoder和decoder网络结构，所以整体网络结构中的encoder和decoder被分成两个部分。在pth模型转换pt模型时，网络的输入和输出均要求是tensor的结构，而示例中encoder部分采用多尺度特征图给到decoder来做后续操作，所以encoder网络的输出为一个list结构，如下代码所示：

```python
    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]
```



所以这里需要将encoder和decoder的网络结构整合到一起，这里采用ResNet101作为encoder的网络结构，UperNet作为decoder的网络结构，合并网络后，即修改models文件夹下的models.py文件，修改后为：

```python
import torch
import torch.nn as nn
from . import resnet, resnext, mobilenet, hrnet
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.net = net
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.net(feed_dict['img_data'], return_feature_maps=True)
            else:
                pred = self.net(feed_dict['img_data'], return_feature_maps=True)

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            pred = self.net(feed_dict['img_data'], segSize=segSize)
            return pred

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_net(weights='', fc_dim=512, num_class=150, use_softmax=False):

        pretrained = True if len(weights) == 0 else False
        orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)

        net = Resnet_UPerNet(orig_resnet, num_class=num_class,
                             fc_dim=fc_dim, use_softmax=use_softmax)

        if len(weights) > 0:
            print('Loading weights for net')
            net.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

        return net



def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class Resnet_UPerNet(nn.Module):
    def __init__(self, orig_resnet, num_class=150, fc_dim=4096, use_softmax=True,
                 pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048),
                 fpn_dim=256):

        super(Resnet_UPerNet, self).__init__()

        ########## encoder ##########
        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        ########## decoder ##########
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, x, segSize=None):

        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)

        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x

```

以上图像预处理opencv化和网络模型结构修改完后，需要进行重新训练，得到修改后的模型

## 模型转化

本文中模型转换主要参考：https://blog.csdn.net/hiteryang/article/details/105575307中模型转化的思路，需要注意的是模型的输入和输出必须要是tensor，否则会出现转化失败的情况

## C++数据预处理、前向传播及后处理代码开发

目前python训练的模型已经转换完毕，后面就是需要对照python前向转播的代码进行C++化，这里主要涉及到数据预处理、前向传播及后处理三个部分的代码

### 数据预处理

在第一节整体工程代码结构中，单张测试脚本test_img.py里的数据预处理代码为：

```python
def data_preprocess_opencv(img_path):

    def round2nearest_multiple(x, p):
        return ((x - 1) // p + 1) * p

    def normalize(tensor, mean, std):
        if not torch.is_tensor(tensor) and tensor.ndimension() == 3:
            raise TypeError('tensor is not a torch image.')
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def img_transform(img):

        # 0-255 to 0-1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = normalize(torch.from_numpy(img.copy()), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_height, ori_width = img.shape[:2]
    this_short_size = 600
    scale = min(this_short_size / float(min(ori_height, ori_width)),
                1000 / float(max(ori_height, ori_width)))
    target_height, target_width = int(ori_height * scale), int(ori_width * scale)

    # to avoid rounding in network
    target_width = round2nearest_multiple(target_width, 32)
    target_height = round2nearest_multiple(target_height, 32)

    # resize images
    img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # # image transform, to torch float tensor 3xHxW
    img_resized = img_transform(img_resized)
    img_resized = torch.unsqueeze(img_resized, 0)

    return img_resized
```

这里主要参考：https://zhuanlan.zhihu.com/p/141401062文章中的对齐方式，但是按照该文章中的写法，最终C++的前向推理时会报段错误，所以这里经过了一些调整，经过转换后的C++代码为：

```C++
int round2nearest_multiple(int x, int p){
    return (floor((x - 1) / p + 1)) * p;
}

torch::Tensor data_preprocess(string img_path, torch::Device device)
{
    cv::Mat input_image = cv::imread(img_path, cv::IMREAD_COLOR);

    if (input_image.empty() || !input_image.data){
        std::cout << "read image fail" << std::endl;
    };

    cv::Mat image;
    cv::cvtColor(input_image, image, cv::COLOR_BGR2RGB);
    int ori_height = image.rows;
    int ori_width = image.cols;

    int this_short_size = 600;
    float scale = std::min(this_short_size / float(min(ori_height, ori_width)),
                           1000 / float(std::max(ori_height, ori_width)));
    int target_height = int(ori_height * scale);
    int target_width = int(ori_width * scale);

    // to avoid rounding in network
    target_width = round2nearest_multiple(target_width, 32);
    target_height = round2nearest_multiple(target_height, 32);

    cv::Size resize_scale = cv::Size(target_width, target_height);
    cv::resize(image, image, resize_scale, 0, 0, cv::INTER_LINEAR);

    /*
    //按照https://zhuanlan.zhihu.com/p/141401062中的写法，预测阶段会出现段错误，暂时没有找到原因
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols,3});
    tensor_image = tensor_image.permute({0,3,1,2});
     */

    torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte).to(device);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);

    tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
    tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
    tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);

    return tensor_image;
}
```



## 前向传播

在第一节整体工程代码结构中，单张测试脚本test_img.py里的前向传播代码为：

```
img_ori = cv2.imread(img_path)
ori_height, ori_width = img_ori.shape[:2]

img = data_preprocess(img_path)
scores = torch.zeros(1, cfg.DATASET.num_class, ori_height, ori_width)

feed_dict['img_data'] = img

# forward pass
pred_tmp = segmentation_module(feed_dict, segSize=(ori_height, ori_width))

scores = scores + pred_tmp
_, pred = torch.max(scores, dim=1)
pred = as_numpy(pred.squeeze(0).cpu())
```

转化后的C++代码为：

```C++
cv::Mat input_image = cv::imread(img_path, cv::IMREAD_COLOR);
int ori_height = input_image.rows;
int ori_width = input_image.cols;
torch::Tensor img_var = data_preprocess(input_image, device);

torch::Tensor pred_tmp = module.forward({ img_var }).toTensor();

pred_tmp = torch::nn::functional::interpolate(pred_tmp, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ori_width, ori_height})));
torch::Tensor scores = torch::zeros({1, 2, ori_width, ori_height});
torch::Tensor final_scores = scores + pred_tmp;
std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(final_scores, 1);
auto max_1= std::get<0>(max_classes);
auto max_index= std::get<1>(max_classes);

torch::Tensor final_result = max_index.squeeze().to(torch::kCPU);
cv::Mat final_mat(cv::Size(ori_width, ori_height), CV_8U, final_result.data_ptr());
```

## opencv和libtorch联合编译

官网编译指引地址：https://pytorch.org/cppdocs/installing.html

按照以上编译流程进行操作即可，这里主要存在的问题是opencv和libtorch联合编译的问题，也即是Cmakelist.txt文件的写法，这里我opencv和libtorch都是安装在自定义的目录下：

```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

# 用下面这种opencv的写法编译不通过，找不到opencv的包，暂时还不清楚为啥
# SET(OpenCV_DIR /path/to/your/opencv/root)

SET(CMAKE_PREFIX_PATH /path/to/your/opencv/root)
SET(Torch_DIR /path/to/your/libtorch/share/cmake/Torch)
# 下面是我opencv需要cuda8.0，所以加上
SET(CUDA_TOOLKIT_ROOT_DIR /path/to/your/cuda)


find_package(OpenCV REQUIRED ${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

if(NOT OpenCV_FOUND)
message(FATAL_ERROR "OpenCV Not Found!")
endif(NOT OpenCV_FOUND)

message(STATUS "OpenCV library status:")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")


find_package(Torch REQUIRED ${Torch_DIR})

if(NOT Torch_FOUND)
message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS " libraries: ${TORCH_LIBRARIES}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(seg_main seg_main.cpp)
target_link_libraries(seg_main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET seg_main PROPERTY CXX_STANDARD 14)
```



大家有什么问题，或者有什么不对的地方可以在博客或者issues里留言：
博客：https://blog.csdn.net/sinat_28731575/article/details/110754888

