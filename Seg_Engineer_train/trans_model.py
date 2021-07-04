# coding:utf-8

# System libs
import os
import argparse
from distutils.version import LooseVersion

# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import time
from os.path import join as opj
import shutil
import pdb
import cv2
from PIL import Image
from tqdm import tqdm

# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy

curr = os.path.dirname(os.path.realpath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class RegionSegmentation:

    def __init__(self, model_path,
                 img,
                 arch_encoder="resnet101",
                 arch_decoder="upernet",
                 fc_dim=2048,
                 img_size=600,
                 img_max_size=1000,
                 padding_constant=32,
                 gpu_flag=True):

        if gpu_flag:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.colors_binary = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        self.colors_mask = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        self.img_size = img_size
        self.img_max_size = img_max_size
        self.padding_constant = padding_constant

        feed_dict = {}
        img_ori = img.convert('RGB')
        ori_height, ori_width = np.array(img_ori).shape[:2]

        img_tensor = self.data_preprocess(img)
        # feed_dict['img_data'] = img_tensor
        # example = torch.rand(1, 3, 544, 960)

        # absolute paths of model weights
        encoder_path = opj(model_path, 'encoder.pth')
        decoder_path = opj(model_path, 'decoder.pth')

        assert os.path.exists(encoder_path) and \
               os.path.exists(decoder_path), \
            model_path + ": weights does not exists!"

        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=arch_encoder,
            fc_dim=fc_dim,
            weights=encoder_path)
        net_decoder = ModelBuilder.build_decoder(
            arch=arch_decoder,
            fc_dim=fc_dim,
            num_class=2,
            weights=decoder_path,
            use_softmax=True)

        # forward pass
        # pdb.set_trace()
        net_encoder.eval()
        net_decoder.eval()

        encoder_trace = torch.jit.trace(net_encoder, img_tensor)
        encoder_trace.save("model_folder/encoder_model_trace.pt")

        conv_out_1, conv_out_2, conv_out_3, conv_out_4 = net_encoder(img_tensor)
        decoder_trace = torch.jit.trace(net_decoder, (conv_out_1, conv_out_2, conv_out_3, conv_out_4))
        decoder_trace.save("model_folder/decoder_model_trace.pt")

        # crit = nn.NLLLoss(ignore_index=-1)
        # self.segmentation_module = \
        #     SegmentationModule(net_encoder, net_decoder, crit).to(self.device)

    def data_preprocess(self, img):
        def round2nearest_multiple(x, p):
            return ((x - 1) // p + 1) * p

        def img_transform(img):
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

            # 0-255 to 0-1
            img = np.float32(np.array(img)) / 255.
            img = img.transpose((2, 0, 1))
            img = normalize(torch.from_numpy(img.copy()))
            return img

        img = img.convert('RGB')
        ori_width, ori_height = img.size
        this_short_size = self.img_size
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.img_max_size / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = round2nearest_multiple(target_width, self.padding_constant)
        target_height = round2nearest_multiple(target_height, self.padding_constant)

        # resize images
        img_resized = img.resize((target_width, target_height), Image.NEAREST)

        # image transform, to torch float tensor 3xHxW
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)

        return img_resized

    def visualize_result(self, pred):

        # print predictions in descending order
        pred = np.int32(pred)

        # colorize prediction
        pred_color_binary = colorEncode(pred, self.colors_binary).astype(np.uint8)
        # pred_color_binary = Image.fromarray(pred_color_binary)
        return pred_color_binary

    def predict(self, img):

        self.segmentation_module.eval()
        feed_dict = {}

        img_ori = img.convert('RGB')
        ori_height, ori_width = np.array(img_ori).shape[:2]

        img_tensor = self.data_preprocess(img)
        scores = torch.zeros(1, 2, ori_height, ori_width).to(self.device)
        feed_dict['img_data'] = img_tensor.to(self.device)

        # scores = async_copy_to(scores, gpu=0)
        # feed_dict = async_copy_to(feed_dict, gpu=0)

        # forward pass
        pred_tmp = self.segmentation_module(feed_dict, segSize=(ori_height, ori_width))

        scores = scores + pred_tmp
        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        pred_color_binary = self.visualize_result(pred)
        return pred_color_binary

    def trans_model(self, img):

        self.segmentation_module.eval()
        feed_dict = {}

        img_ori = img.convert('RGB')
        ori_height, ori_width = np.array(img_ori).shape[:2]

        img_tensor = self.data_preprocess(img)
        scores = torch.zeros(1, 2, ori_height, ori_width).to(self.device)
        feed_dict['img_data'] = img_tensor.to(self.device)

        # forward pass
        traced_script_module = torch.jit.trace(self.segmentation_module, feed_dict)
        traced_script_module.save("model_folder/seg_model_trace.pt")


def mkr(image_dir_out):
    if not os.path.exists(image_dir_out):
        print("we create {} dir".format(image_dir_out))
        os.makedirs(image_dir_out)
    else:
        shutil.rmtree(image_dir_out)
        os.makedirs(image_dir_out)

if __name__ == '__main__':

    model_path = "/path/to/model/file"
    img = Image.open("/path/to/img/file")
    regionSeg = RegionSegmentation(model_path, img, gpu_flag=True)
    # regionSeg.trans_model(img)


