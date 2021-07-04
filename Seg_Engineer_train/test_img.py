# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.io import loadmat
import csv
import time
from os.path import join as opj
import shutil
import pdb
import cv2

# Our libs
from mit_semseg.models.model_build import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

curr = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colors_binary = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
colors_mask = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
names = {1: "background", 2: "water bill"}


def visualize_result(data, pred, cfg, mask_dir, binary_dir):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color_binary = colorEncode(pred, colors_binary).astype(np.uint8)

    # mask img
    pred_color_mask = colorEncode(pred, colors_mask, mode="BGR").astype(np.uint8)
    pred_color = 0.7 * pred_color_mask + img

    # aggregate images and save
    # im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(pred_color_binary).save(
        os.path.join(binary_dir, img_name.replace('.jpg', '.png')))
    cv2.imwrite(os.path.join(mask_dir, img_name), pred_color)

def data_preprocess(img_path):

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

    img = Image.open(img_path).convert('RGB')
    ori_width, ori_height = img.size
    this_short_size = cfg.DATASET.imgSizes
    scale = min(this_short_size / float(min(ori_height, ori_width)),
                cfg.DATASET.imgMaxSize / float(max(ori_height, ori_width)))
    target_height, target_width = int(ori_height * scale), int(ori_width * scale)

    # to avoid rounding in network
    target_width = round2nearest_multiple(target_width, cfg.DATASET.padding_constant)
    target_height = round2nearest_multiple(target_height, cfg.DATASET.padding_constant)

    # resize images
    img_resized = img.resize((target_width, target_height), Image.NEAREST)

    # image transform, to torch float tensor 3xHxW
    img_resized = img_transform(img_resized)
    img_resized = torch.unsqueeze(img_resized, 0)

    return img_resized

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

def test(segmentation_module, img_path_list, gpu, mask_dir, binary_dir):

    segmentation_module.eval()

    feed_dict = {}
    for img_path in tqdm(img_path_list):
        # img_ori = Image.open(img_path).convert('RGB')
        img_ori = cv2.imread(img_path)
        ori_height, ori_width = img_ori.shape[:2]

        img = data_preprocess_opencv(img_path)
        scores = torch.zeros(1, cfg.DATASET.num_class, ori_height, ori_width)
        # scores = async_copy_to(scores, gpu)

        feed_dict['img_data'] = img
        # feed_dict = async_copy_to(feed_dict, gpu)

        # forward pass
        # torch.cuda.synchronize(device)
        start = time.time()
        pred_tmp = segmentation_module(feed_dict, segSize=(ori_height, ori_width))
        # torch.cuda.synchronize(device)
        end = time.time()
        print(" -- Time: {:.2f}ms".format((end - start) * 1000))

        scores = scores + pred_tmp
        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result(
            (np.array(img_ori), img_path),
            pred,
            cfg, mask_dir, binary_dir
        )


def mkr(image_dir_out):
    if not os.path.exists(image_dir_out):
        print("we create {} dir".format(image_dir_out))
        os.makedirs(image_dir_out)
    else:
        shutil.rmtree(image_dir_out)
        os.makedirs(image_dir_out)

def main_entrance(args, imgs_content, mask_dir, binary_dir, cfg_file, model_path, checkpoint_name):

    ###################### 初始加载 ######################
    mkr(mask_dir)
    mkr(binary_dir)

    cfg.merge_from_file(cfg_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)  # TODO
    # logger.info("Loaded configuration file {}".format(cfg_file))
    # logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    weights_path = opj(model_path, 'seg_' + checkpoint_name)

    # generate testing image list
    if os.path.isdir(imgs_content):
        imgs = find_recursive(imgs_content)
    else:
        imgs = [imgs_content]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]
    img_path_list = [x for x in imgs]


    ####################### 主流程 ################################
    # torch.cuda.set_device(args.gpu)

    # Network Builders
    seg_net = ModelBuilder.build_net(
        weights=weights_path,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        use_softmax=True
    )

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(seg_net, crit)

    # # Dataset and Loader
    # dataset_test = TestDataset(
    #     cfg.list_test,
    #     cfg.DATASET)
    # loader_test = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=cfg.TEST.batch_size,
    #     shuffle=False,
    #     collate_fn=user_scattered_collate,
    #     num_workers=5,
    #     drop_last=True)

    # segmentation_module.cuda()

    # Main loop
    test(segmentation_module, img_path_list, args.gpu, mask_dir, binary_dir)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        default="",
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    args = parser.parse_args()

    imgs_content = "/path/to/img/file/dir"
    results_dir = "/path/to/output/dir"
    mask_dir = "/path/to/output/mask/dir"
    binary_dir = "/path/to/output/binary/dir"
    cfg_file = "/path/to/config/yaml/file"
    model_path = "/path/to/model"
    checkpoint_name = "model_name"

    main_entrance(args, imgs_content, mask_dir, binary_dir, cfg_file, model_path, checkpoint_name)




