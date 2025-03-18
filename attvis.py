import json

import cv2
import matplotlib as mpl
import pylab
from cv2 import imread
from scipy.stats import entropy
from tqdm import tqdm

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # for Windows
import skimage.transform
import random
from data_pro import ImageDetectionsField, TextField, RawField, PixelField, ImageDetectionsField2
from data_pro import COCO, DataLoader
import evaluation
from models.diffnet_pro import Difnet, DifnetEncoder
from models.diffnet_pro.decoders2 import DifnetDecoder
from models.transformer import TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Transformer

import torch
import argparse
import os
import pickle
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_cap(model, dataloader, text_field, outfile):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    outs = {}
    plt.figure()
    with tqdm(desc='test', unit='it', total=len(dataloader)) as pbar:
        for it, ((images, pixels, img_path, region), caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            pixels = pixels.to(device)
            region = region.to(device)
            image_id = int(img_path[0].split('_')[-1].split('.')[0])
            img = cv2.resize(imread(img_path[0]),(224,224))
            with torch.no_grad():
                out, log_probs, atts = model.beam_search2(images, pixels, region, 20, text_field.vocab.stoi['<eos>'], 5,
                                                          out_size=1)
            caps_gen = text_field.decode(out, join_words=False)

            num_words = len(caps_gen[0])
            columns = num_words + 1  # 额外一列用于最终原图
            rows = 2  # 每个单词 + 注意力图各占一行

            plt.figure(figsize=(columns * 2, 4))  # 增大图片以适应长句子

            for t in range(num_words):
                plt.subplot(rows, columns, t + 1)
                plt.axis('off')
                plt.text(0.5, 0.5, f'{caps_gen[0][t]}',
                         ha='center', va='center', fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.8))
                # 处理注意力张量
                att_t = atts[t].squeeze(0)  # 形状变为 (8, 1, 49)
                att_mean = att_t.mean(dim=0)  # 沿注意力头维度求平均 → (1,49)
                att_2d = att_mean.view(7, 7).cpu().numpy()  # 转换为7x7 numpy数组
                resized_att = skimage.transform.resize(att_2d, img.shape[:2],
                                                           mode='reflect',
                                                           anti_aliasing=True)

                # 叠加显示热力图
                plt.subplot(rows, columns, columns + t + 1)
                plt.axis('off')
                plt.imshow(img)
                plt.imshow(resized_att, alpha=0.6, cmap='jet')  # 调整 colormap
                plt.colorbar()

            # 句子的最后，放置原图
            plt.subplot(rows, columns, columns)
            plt.axis('off')
            plt.imshow(img)

            plt.tight_layout()
            plt.savefig(f"caption_image/caption_{image_id}.png")
            plt.close()
            pbar.update()


    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    json.dump(outs, outfile)
    return scores


def all_inp_entropy(data, pos=None):
    res = []
    for i in range(len(data)):
        if not (np.isnan(data[i]['inp_lrp'])).any():
            res_ = np.sum(abs(data[i]['inp_lrp']), axis=-1)
            try:
                if pos is None:
                    res += [entropy(abs(data[i]['inp_lrp'][p]) / res_[p]) for p in range(data[i]['inp_lrp'].shape[0])]
                else:
                    res.append(entropy(abs(data[i]['inp_lrp'][pos]) / res_[pos]))
            except Exception:
                pass
    # print(res)
    return res


def avg_lrp_by_pos(data, seg='inp'):
    count = 0
    res = np.zeros(20)
    for i in range(len(data)):
        if not (np.isnan(data[i][seg + '_lrp'])).any():
            d = np.sum(data[i][seg + '_lrp'], axis=-1)
            d = np.pad(d, (0, 20 - d.shape[0]), 'constant', constant_values=(0, 0))
            res += d
            count += 1
    res /= count
    return res


def show_source2target(all_data, labels, save_img_contribution):
    # spectral_map = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    # color = spectral_map[-1]
    fig = plt.figure(figsize=(7, 6), dpi=100)
    for i, (data, label) in enumerate(zip(all_data, labels)):
        res = avg_lrp_by_pos(data, seg='inp')[1:]
        plt.plot(range(2, 9 + 2), res[:9], label=label)

    # res = avg_lrp_by_pos(data, seg='inp')[1:]
    # res1 = avg_lrp_by_pos(data1, seg='inp')[1:]
    # res2 = avg_lrp_by_pos(data2, seg='inp')[1:]
    # plt.plot(range(2, len(res) + 2), res, 'bv-', label='Baseline w/o Enc: extra skip connection')
    # plt.plot(range(2, len(res) + 2), res1, 'g^--', label='Baseline')
    # plt.plot(range(2, len(res) + 2), res2, 'ro-', label='Ours')
    # plt.plot(range(2, 9+2), res[:9], 'bv-', label='Baseline w/o Enc: extra skip connection')
    # plt.plot(range(2, 9+2), res1[:9], 'g^--', label='Baseline')
    # plt.plot(range(2, 9+2), res2[:9], 'ro-', label='Ours')
    # plt.plot(range(2, len(res)-3), res[:14], lw=2., color='b')
    # plt.plot(range(2, len(res) - 3), res1[:14], lw=2., color='g')
    # plt.plot(range(2, len(res) - 3), res2[:14], lw=2., color='r')
    # plt.scatter(range(2, len(res)-3), res[:14], lw=3.0, color=color)
    # plt.scatter(range(2, len(res) - 3), res1[:14], lw=3.0, color=color)
    # plt.scatter(range(2, len(res) - 3), res2[:14], lw=3.0, color=color)

    plt.xlabel("target token position", size=20)
    plt.ylabel("visual contribution", size=20)
    plt.yticks(size=20)
    plt.xticks([2, 5, 10, 11], size=17)
    plt.title('visual ⟶ target(k)', size=20)
    plt.grid()
    plt.legend(fontsize=13)
    pylab.savefig(save_img_contribution, bbox_inches='tight')


def show_entropy(data, data1, data2, pictdir):
    # entropy
    color = 'b'
    fig = plt.figure(figsize=(7, 6), dpi=100)
    res = [np.mean(all_inp_entropy(data, pos=pos)) for pos in range(20)]
    res1 = [np.mean(all_inp_entropy(data1, pos=pos)) for pos in range(20)]
    res2 = [np.mean(all_inp_entropy(data2, pos=pos)) for pos in range(20)]
    # res = [np.mean(all_inp_entropy(data))]
    # plt.plot(range(1, len(res) + 1), res, lw=2., color=color)
    # plt.scatter(range(1, len(res) + 1), res, lw=3.0, color=color)
    plt.plot(range(2, len(res) + 2), res, 'bv-', label='Baseline w/o Enc: extra skip connection')
    plt.plot(range(2, len(res) + 2), res1, 'g^--', label='Baseline')
    plt.plot(range(2, len(res) + 2), res2, 'ro-', label='Ours')

    plt.xlabel("target token position", size=25)
    plt.ylabel("entropy", size=25)
    plt.yticks(size=20)
    plt.xticks([1, 5, 10, 15, 17], size=17)
    plt.title('Entropy of visual contributions', size=25)
    plt.grid()
    plt.legend()
    # pylab.savefig(pictdir + 'YOUR FNAME', bbox_inches='tight')
    pylab.savefig(pictdir, bbox_inches='tight')


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='DIFNetpro')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', default=True, action='store_true')
    parser.add_argument('--features_path', type=str,
                        default='D:/BaiduNetdiskDownload/coco_grid_feats/combined_detections.hdf5')
    parser.add_argument('--region_path', type=str,
                        default='D:/BaiduNetdiskDownload/aaai21_DLCT/coco_all_align/coco_all_align.hdf5')
    parser.add_argument('--pixel_path', type=str, default='dataset/segmentations')
    parser.add_argument('--annotation_folder', type=str, default='dataset/annotations')
    parser.add_argument('--logs_folder', type=str, default='./output/tensorboard_logs')
    parser.add_argument('--model_path', type=str, default='./output/saved_transformer_models')
    parser.add_argument('--exp_name', type=str, default='transformer_grid_original')
    parser.add_argument('--mode', type=str, default='difnet_PRO', choices=['difnet', 'difnet_PRO'])
    parser.add_argument('--out_path', type=str, default='./output/output_lrp')
    args = parser.parse_args()
    print(args)

    # Pipeline for image grid
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)
    # Pipeline for image regions
    detect_field = ImageDetectionsField2(detections_path=args.region_path, max_detections=49, load_in_tmp=False)
    # Pipeline for pixel
    pixel_field = PixelField(pixel_path=args.pixel_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True,
                           nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, pixel_field, detect_field, 'dataset/coco/images',
                   args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    encoder = DifnetEncoder(1, 2, 2, 0, attention_module=ScaledDotProductAttention)
    decoder = DifnetDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Difnet(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    # fname = "output/saved_transformer_models/DIFNetpro_best.pth"
    fname = "copy_models/DIFNet_lrp.pth"
    data = torch.load(fname)
    model.load_state_dict(data['state_dict'], strict=False)

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(),
                                                       'pixel': pixel_field, 'detect': detect_field})

    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size)
    # Validation scores
    out_file = open(os.path.join(args.out_path, args.exp_name + '.json'), 'w')
    scores = predict_cap(model, dict_dataloader_test, text_field, out_file)
    out_file.close()

    save_img_contribution = os.path.join(args.out_path, '{}.jpg'.format('contribution'))
    save_img_entropy = os.path.join(args.out_path, '{}.jpg'.format('entropy'))

    all_data = []
    for exp in args.exp_name:
        fname = '{}_result.pkl'.format(exp)
        data = pickle.load(open(os.path.join(args.out_path, fname), 'rb'))
        all_data.append(data)
    show_source2target(all_data, args.exp_name, save_img_contribution)
