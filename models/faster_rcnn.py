from .modules.mask_rcnn import FasterRCNN

def create_model(data, config_of_data, num_classes=80, backbone='resnet-50-c4',
                 share_features=True, **kwargs):
    return FasterRCNN(backbone=backbone, num_classes=num_classes)
