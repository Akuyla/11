# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'backbone': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'use_p2': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channels_list': [64, 128, 256],
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'backbone': 'resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'use_p2': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channels_list': [512, 1024, 2048],
    'out_channel': 256
}

cfg_resnest50 = {
    'name': 'ResNeSt50',
    'backbone': 'resnest50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'use_p2': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channels_list': [512, 1024, 2048],
    'out_channel': 256
}

cfg_re50_p2 = {
    'name': 'Resnet50_P2',
    'backbone': 'resnet50',
    'min_sizes': [[8, 16], [32, 64], [128, 256], [512, 768]],
    'steps': [4, 8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'use_p2': True,
    'return_layers': {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4},
    'in_channels_list': [256, 512, 1024, 2048],
    'out_channel': 256
}

# cfg_resnest50_p2 = {
#     'name': 'ResNeSt50_P2',
#     'min_sizes': [[8, 16], [32, 64], [128, 256], [512, 768]],
#     'steps': [4, 8, 16, 32],
#     'variance': [0.1, 0.2],
#     'clip': False,
#     'loc_weight': 2.0,
#     'gpu_train': True,
#     'batch_size': 24,
#     'ngpu': 4,
#     'epoch': 100,
#     'decay1': 70,
#     'decay2': 90,
#     'image_size': 840,
#     'pretrain': True,
#     'use_p2': True,
#     'return_layers': {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4},
#     'in_channels_list': [256, 512, 1024, 2048],
#     'out_channel': 256
# }
cfg_resnest50_p2 = {
    'name': 'ResNeSt50_P2',
    'backbone': 'resnest50',
    'min_sizes': [[8, 16], [32, 64], [128, 256], [512, 768]],
    'steps': [4, 8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 800,

    'pretrain': True,
    'use_p2': True,
    'return_layers': {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4},
    'in_channels_list': [256, 512, 1024, 2048],
    'out_channel': 256,

        # 优化器设置
    'optimizer': 'SGD',
    'lr': 0.0005,                # 与ResNet50一致
    'weight_decay': 5e-4,
    'momentum': 0.9,             # 补充动量（原未写，但代码中通常使用）
    
    # 混合精度训练（关键！节省显存）
    'amp': True,
    
    'num_workers': 10,           # 数据加载线程数
    'pin_memory': True,          # 加速数据传输
    'save_interval': 5,          # 每5个epoch保存一次
    'keep_checkpoints': 3,       # 保留最近3个checkpoint
}
