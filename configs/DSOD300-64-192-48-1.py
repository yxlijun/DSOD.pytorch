model = dict(
    type='DSOD300-64-192-48-1',
    input_size=300,
    init_net=True,
    rgb_means=(104, 117, 123),
    init_features=64,
    block_config=[6, 8, 8, 8],
    bottleneck_1x1_num=192,
    growth_rate=48,
    num_classes=None,
    anchor_config=dict(
        feature_maps=[38, 19, 10, 5, 3, 1],
        steps=[8, 16, 32, 64, 100, 300],
        min_sizes=[30, 60, 111, 162, 213, 264],
        max_sizes=[60, 111, 162, 213, 264, 315],
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        anchor_nums=[4, 6, 6, 6, 4, 4],
    ),
    p=0.6,
    save_epochs=20,
    weights_save='weights/'
)

train_cfg = dict(
    cuda=True,
    per_batch_size=16,
    lr=1e-3,
    gamma=0.1,
    end_lr=1e-6,
    step_lr=dict(
        COCO=[200, 250, 300, 360],
        VOC=[500, 600, 700, 800]
    ),
    print_epochs=10,
    num_workers=8,
)

test_cfg = dict(
    cuda=True,
    topk=0,
    iou=0.45,
    soft_nms=True,
    score_threshold=0.1,
    keep_per_class=50,
    save_folder='eval',
)

loss = dict(overlap_thresh=0.5,
            prior_for_matching=True,
            bkg_label=0,
            neg_mining=True,
            neg_pos=3,
            neg_overlap=0.5,
            encode_target=False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC=dict(
        train_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets=[('2007', 'test')],
    ),
    COCO=dict(
        train_sets=[('2014', 'train'), ('2014', 'valminusminival')],
        eval_sets=[('2014', 'minival')],
        test_sets=[('2015', 'test-dev')],
    )
)

import os
home = os.path.expanduser("~")
VOCroot = os.path.join(home, "data/VOCdevkit/")
COCOroot = os.path.join(home, "data/coco/")
