import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from nwd import NuclearWassersteinDiscrepancy
from grl import WarmStartGradientReverseLayer

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, num_classes=8, bottleneck_width=256):
        super(Classifier, self).__init__()
        # 加载预训练的ResNet模型
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # 添加瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(resnet.fc.in_features, bottleneck_width),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 分类层
        # self.classifier_layer = nn.Sequential(
        #     nn.Linear(resnet.fc.in_features, bottleneck_width),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(bottleneck_width, num_classes)
        # )
        # self.classifier_layer = nn.Sequential(
        #     nn.Linear(resnet.fc.in_features, num_classes)
        # )
        self.classifier_layer = nn.Sequential(
            nn.Linear(bottleneck_width, num_classes)
        )

    #---------------ToAlign-----------------------
    def _get_toalign_weight(self, f, labels):
        # weight是分类层的第四层中获取权重
        # labels是一个包含类别索引的张量，其长度等于批次大小batch_size，表示每个样本的类别
        # detach方法被调用来从当前计算图中分离这些权重，使它们不会在反向传播中更新
        # w的形状是[batch_size, c]
        w = self.classifier_layer[0].weight[labels].detach()
        eng_org = (f**2).sum(dim=1, keepdim=True)
        eng_aft = ((f*w)**2).sum(dim=1, keepdim=True)
        scalar = (eng_org / eng_aft).sqrt()
        w_pos = w * scalar

        return w_pos


    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        f_reduced = self.bottleneck(f)
        output = self.classifier_layer(f_reduced)
        # output = self.classifier_layer(f)
        return f_reduced, output
        # return f, output

class DALN(nn.Module):
    def __init__(self, num_classes, bottleneck_width=256):
        super(DALN, self).__init__()
        self.classifier = Classifier(num_classes, bottleneck_width)
        self.discrepancy = NuclearWassersteinDiscrepancy(self.classifier.classifier_layer)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x_s, labels_s, x_t, trade_off_lambda=1.0, toalign=True):
        # 处理源域数据
        f_s, y_s = self.classifier(x_s)
        # cls_loss = self.ce_loss(y_s, labels_s)  # 分类损失

        # 计算ToAilgn权重并调整源域特征
        if toalign:
            w_pos_s = self.classifier._get_toalign_weight(f_s, labels_s)
            f_s = f_s * w_pos_s
            # y_s = self.classifier.classifier_layer(f_s)

        cls_loss = self.ce_loss(y_s, labels_s)  # toalign之后的分类损失

        # 处理目标域数据
        f_t, _ = self.classifier(x_t)

        # 合并源域和目标域特征
        f_combined = torch.cat((f_s, f_t), dim=0)  # 相当于合并了源域的特征以及目标域的特征
        
        # 计算域适应损失
        discrepancy_loss = -self.discrepancy(f_combined)
        transfer_loss = discrepancy_loss * trade_off_lambda

        # 总损失
        # print(cls_loss, transfer_loss)
        loss = cls_loss + transfer_loss
        return loss
