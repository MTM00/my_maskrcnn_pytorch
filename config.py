# this is used for storing configurations of datasets & models

datasets = {
    'coco-trainval35k-minival': {
        'num_classes': 80,
        'scale_size': 800,
        'train_split': 'annotations/instances_trainval35k2014.json',
        'val_split': 'annotations/instances_minival2014.json',
        'test_split': 'annotations/image_info_test-dev2015.json',
    },
    'coco-train-minival': {
        'num_classes': 80,
        'scale_size': 800,
        'train_split': 'annotations/instances_train2014.json',
        'val_split': 'annotations/instances_minival2014.json',
        'test_split': 'annotations/image_info_test-dev2015.json',
    },
    'coco-train-val': {
        'num_classes': 80,
        'scale_size': 800,
        'train_split': 'annotations/instances_train2014.json',
        'val_split': 'annotations/instances_val2014.json',
        'test_split': 'annotations/image_info_test-dev2015.json',
    },
    'coco-debug': {
        'num_classes': 80,
        'scale_size': 600,
        'train_split': 'annotations/instances_minival2014.json',
        'val_split': 'annotations/instances_minival2014.json',
        'test_split': 'annotations/image_info_test-dev2015.json',
    },
}
