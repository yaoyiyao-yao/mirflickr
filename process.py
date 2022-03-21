from os.path import join
from os import listdir
import numpy as np
import scipy.io as sio
from imageio import imread
from skimage.transform import resize
from keras.applications.vgg19 import VGG19
from keras.models import Model
import tqdm

P = "E:/workspace/dataset/mirflickr_dataset"
BASE = join(P, 'mirflickr')
IMG_P = BASE  # image 路径
TXT_P = join(BASE, 'meta/tags')  #  text 路径
LAB_P = join(P, 'mirflickr25k_annotations_v080')  # label 路径
COM_TAG_F = join(BASE, 'doc/common_tags.txt')  # common tags
N_DATA = 25000

# 加载 VGG 19
vgg = VGG19(weights='imagenet')
# vgg.summary()
vgg = Model(vgg.input, vgg.get_layer('fc2').output)  # 取 4096-D feature 输出
vgg.summary()

# image 文件列表
fs_img = [f for f in listdir(IMG_P) if '.jpg' in f]
key_img = lambda s: int(s.split('.jpg')[0].split('im')[-1])
fs_img = sorted(fs_img, key=key_img)  # 按标号排序
fs_img = [join(IMG_P, f) for f in fs_img]

N_IMG = len(fs_img)
print("#images:", N_IMG)

# 提取 image 特征
all_img = []
for i in tqdm.trange(10):
    im = imread(fs_img[i])
    im = resize(im, (224, 224, 3))
    im = np.expand_dims(im, axis=0)
    all_img.append(vgg.predict(im))

all_img = np.vstack(all_img)
print("images shape:", all_img.shape)

# 保存
np.save(join(P, "images.npy"), all_img)


# 处理 common tags
tag_idx, idx_tag = {}, {}
cnt = 0
with open(COM_TAG_F, 'r', encoding='UTF-8') as f:
    for line in f:
        line = line.split()
        tag_idx[line[0]] = cnt
        idx_tag[cnt] = line[0]
        cnt += 1
DIM_TXT = len(tag_idx.keys())
print("text dim:", DIM_TXT)

# text 文件列表
key_txt = lambda s: int(s.split('.txt')[0].split('tags')[-1])  # 按标号排序
fs_tags = sorted(listdir(TXT_P), key=key_txt)
fs_tags = [join(TXT_P, f) for f in fs_tags]

N_TXT = len(fs_tags)
print("#texts:", N_TXT)

def get_tags(tag_f):
    '''
    读 tag 文件，获取该 sample 的 tags
    '''
    tg = []
    with open(tag_f, 'r', encoding='UTF-8') as f:
        for line in f:
            a = line.strip()
            if a in tag_idx:
                tg.append(a)
    return tg

# 制成 BoW
all_txt = np.zeros((N_TXT, DIM_TXT))
for i in tqdm.trange(N_TXT):
    tag = get_tags(fs_tags[i])
    for s in tag:
        if s in tag_idx:  # 在 common tags 内 print(i, s)
            all_txt[i][tag_idx[s]] = 1
print("texts shape:", all_txt.shape)

# 保存
# np.save(join(P, "texts.npy"), all_txt.astype(np.int8))
all_txt = all_txt.astype(np.uint8)
sio.savemat(join(P, "texts.mat"), {"texts": all_txt}, do_compression=True)


# 处理 Label / Annotation
key_lab = lambda s: s.split('.txt')[0]  # 按类名字典序升序
# label 文件列表
fs_lab = [s for s in listdir(LAB_P) if "README" not in s]
fs_lab = [s for s in fs_lab if "_r1" not in s]  # 这行注掉就是 38 个类
fs_lab = sorted(fs_lab, key=key_lab)

with open(join(P, "class-name-{}.txt".format(len(fs_lab))), "w") as f:
    # 记下 class name 与对应的 ID
    # format: <class name>, <class ID>
    # （用来统一 class 顺序）
    for i, c in enumerate(fs_lab):
        c = key_lab(c)
        f.write("{}, {}\n".format(c, i))

fs_lab = [join(LAB_P, s) for s in fs_lab]
N_CLASS = len(fs_lab)
print("#classes:", N_CLASS)

def sample_of_lab(lab_f):
    """读 annotation 文件，获取属于该类的 samples 标号"""
    samples = []
    with open(lab_f, 'r', encoding='UTF-8') as f:
        for line in f:
            sid = int(line)
            samples.append(sid)
    return samples

# 处理 label
all_lab = np.zeros((N_DATA, N_CLASS))
for i in range(len(fs_lab)):
    samp_ls = sample_of_lab(fs_lab[i])
    for s in samp_ls:
        all_lab[s - 1][i] = 1  # s-th 样本属于 i-th 类
print("labels shape:", all_lab.shape)

# 保存
# np.save(join(P, "labels.{}.npy".format(N_CLASS)), all_lab.astype(np.int8))
all_lab = all_lab.astype(np.uint8)
sio.savemat(join(P, "labels.{}.mat".format(N_CLASS)), {"labels": all_lab}, do_compression=True)

