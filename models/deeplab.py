import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.segmentation import my_load_model

def my_deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, only_feature=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return my_load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, only_feature, **kwargs)

if __name__ == '__main__':
    model = my_deeplabv3_resnet50(num_classes=2,only_feature=True)
    print(model)
    # device = torch.device('cuda:1')
    # model = model.to(device)
    # img = torch.rand((4,3,256,256)).to(device)
    # out = model(img)
    # print(out.shape)