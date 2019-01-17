    p = Augmentor.Pipeline()
    eras = RandomErasing(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(eras)
    bright = RandomBrightness(probability=0.4, min_factor=0.4, max_factor=1.5) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(bright)
    color = RandomColor(probability=0.4, min_factor=0.5, max_factor=0.8) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(color)
    contrast = RandomContrast(probability=0.4, min_factor=0.5, max_factor=0.9) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(contrast)
    
    # 训练时候
    'train': transforms.Compose([
            transforms.Resize((64,280)),
            p.torch_transform(),  # 直接转torch处理方式，训练时增强
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 最好计算自己数据集的
        ]),
