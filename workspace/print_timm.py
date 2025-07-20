import timm

import torchsummary
NUM_FINETUNE_CLASSES = 2
model = timm.create_model(
    "wide_resnet50_2", num_classes=NUM_FINETUNE_CLASSES
)
# print(model)
# # 打印特征层名称
# for name, module in model.named_children():
#     print(name)
torchsummary.summary(model, input_size=(3, 320, 320))  # 假设输入是 3x224x224 的图像

