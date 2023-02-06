# CLIP
import torch.cuda
import torchvision.transforms as ttr
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
import matplotlib as mpl
import clip

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def Clip_transform(n_px):
    return ttr.Compose([
        ttr.Resize(n_px),
        ttr.CenterCrop(n_px),
        # _convert_image_to_rgb,
        ttr.ToTensor(),
        ttr.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def Clipfeatures(net, x):
    x = x.type(net.conv1.weight.dtype)
    for conv, bn in [(net.conv1, net.bn1), (net.conv2, net.bn2), (net.conv3, net.bn3)]:
        x = F.relu(bn(conv(x)))
    # print(1,x.shape)
    x = net.avgpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    # print(2,x.shape)
    x = net.attnpool(x)
    return x


#model, preprocess = clip.load('RN50x4', device)
#print(clip.available_models())


# Billion-scale semi-supervised learning for image classification, https://arxiv.org/abs/1905.00546
def swslfeatures(net, x):
    # See note [TorchScript super()]
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)

    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)

    # print(x.shape)
    # x = net.avgpool(x)
    x = F.avg_pool2d(x, x.shape[2], 1)
    x = torch.flatten(x, 1)
    return x


def swsl_transform(n_px):
    return ttr.Compose([
        ttr.Resize(n_px),
        ttr.CenterCrop(n_px),
        # _convert_image_to_rgb,
        ttr.ToTensor(),
        ttr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# SWSL
torch.hub.list('facebookresearch/semi-supervised-ImageNet1K-models')
model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
model = model.to(device)
model.eval()
