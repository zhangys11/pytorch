import os
import math
from PIL import Image
from tqdm import tqdm
import glob
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
from torchvision.transforms.functional import resize, to_pil_image, normalize
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, ScoreCAM,SSCAM, ISCAM, XGradCAM, LayerCAM

########## Image Preprocessing ############

def crop_image_to_center(image_path, output_path, percent=0.85):
    """
    裁剪图片到中间的百分比区域，并保存到指定路径。
    """
    img = Image.open(image_path)
    width, height = img.size
    new_width = int(width * percent / 2)
    new_height = int(height * percent / 2)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(output_path)

def process_image_folders(input_folder, output_folder, percent=0.95):
    """
    处理特定分类下的所有图片。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if os.path.isfile(input_path) and (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
            crop_image_to_center(input_path, output_path, percent)

def preprocess(data_folders=['train', 'test'], classes=['RottenGrape', 'FreshGrape']):
    """
    遍历指定的训练和测试文件夹及其分类，处理图片。
    """
    for data_folder in data_folders:
        for class_name in classes:
            input_folder = os.path.join(data_folder, class_name)
            output_folder = os.path.join(data_folder, f"{class_name}_cropped")
            print(f"Processing {input_folder}...")
            process_image_folders(input_folder, output_folder)

############## Load Datasets ################

def build_dataloaders(root_dirs = ['train','test'], batch_size = 16):
    
    transform_train = transforms.Compose([
        # # 数据增强，随机裁剪224*224大小 # transforms.RandomResizedCrop(224),    
        transforms.Resize((224,224)),
        # 数据增强，随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
        transforms.ToTensor(),
        # 对每个通道的像素进行标准化，给出每个通道的均值和方差
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
     
    # 验证集的数据预处理
    transform_val = transforms.Compose([
        # 将输入图像大小调整为224*224
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
     
    # 读取训练集并预处理
    train_dataset = datasets.ImageFolder(root=root_dirs[0], transform = transform_train)
    # 读取验证集并预处理
    val_dataset = datasets.ImageFolder(root=root_dirs[1], transform = transform_val)
     
    # 查看图像类别及其对应的索引
    class_dict = train_dataset.class_to_idx
    print(class_dict) 
    # 将类别名称保存在列表中
    class_names = list(class_dict.keys())
     
    # 构造训练集
    train_loader = DataLoader(dataset=train_dataset,  # 接收训练集
                              batch_size=batch_size,  # 训练时每个step处理16张图
                              shuffle=True,           # 打乱每个batch
                              num_workers=0)          # 加载数据时的线程数量，windows环境下只能=0
     
    # 构造验证集
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    return train_loader, val_loader, class_names

######### Model Definitions #############

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16): # reducation ratio = 16
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 定义基本残差块
class BasicResidualBlock(nn.Module):
    '''
    Basic Residual Block
    '''
    expansion = 1  # 残差块输出通道数相对于输入通道数的倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认标准化层为BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicResidualBlock只支持groups=1和base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1在BasicResidualBlock中尚未实现")
        # 卷积层和BN层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample  # 下采样操作
        self.stride = stride  # 步长

    def forward(self, x):
        identity = x  # 保存输入的残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 如果需要，进行下采样

        out += identity  # 残差连接
        out = self.relu(out)

        return out

class CbamBlock(nn.Module):
    '''
    CBAM (convolutional block attention model)
    '''
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CbamBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        # self.softmax = nn.Softmax()
 
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        out = self.ca(out) * out
        out = self.sa(out) * out
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
    '''
    Resnet backbone
    '''
    def __init__(self, block, num_planes=[64,128,256,512], num_blocks = [3,4,6,3], num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64 # input size / kernel size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = {}
        for i,(p,b) in enumerate(zip(num_planes, num_blocks)):
            part = self._make_layer(block, planes=p, blocks=b, stride=1+(i>0))
            self.layers['layer'+str(i+1)] = part
            setattr(self, 'layer'+str(i+1), part)
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 修改这里，使得全连接层的输出维度由num_classes指定
        self.fc = nn.Linear(num_planes[-1] * block.expansion, num_classes)
        # self.softmax = nn.Softmax()
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for _, l in self.layers.items():
            x = l(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.softmax(x)
 
        return x

############### Model Training ##############
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_learning_curve(train_loss, val_loss, train_acc, val_acc):
        
    # 绘制损失和准确率曲线
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

def train_model(model, train_loader, val_loader, weight_path, LR, EPOCHS):
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # 定义使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 把模型转移到GPU上运行
    
    model.to(device)
    
    # 定义交叉熵损失
    loss_function = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 保存准确率最高的一次迭代
    best_acc = 0.0
    for epoch in range(EPOCHS):
        print('-'*30, '\n', f'Epoch: {epoch}/{EPOCHS-1}')
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_correct_cnt = 0  # 初始化训练正确数计数器
        
        # 训练循环
        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = loss_function(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct_cnt += (predicted == labels).sum().item() 
            
        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
    
        # 计算训练准确率
        train_accuracy = 100 * train_correct_cnt / (len(train_loader.dataset))  # 使用数据加载器的数据集长度
        train_accuracies.append(train_accuracy)
    
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct_cnt = 0
    
        # 遍历验证集
        with torch.no_grad():
            for data_test in val_loader:
                test_images, test_labels = data_test
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                outputs = model(test_images)
                loss = loss_function(outputs, test_labels)
                val_running_loss += loss.item()            
                _, predicted = outputs.max(1)  # 不需要使用.cpu()方法
                val_correct_cnt += (predicted.cpu() == test_labels.cpu()).sum().item()  # 将predicted和test_labels移动到CPU上再进行比较
    
        avg_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # 计算验证准确率
        val_accuracy = 100 * val_correct_cnt / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)
        print(f'Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | '
              f'Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%')
    
        # 保存最好的模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), weight_path)
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

    plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies)

    return model

########################## CAM ###########################

def predict_and_visualize_heatmap(model, image_paths='.', weights_path="cbam_best.pth", target_layer='last'):

    if image_paths == '.' or os.path.isdir(image_paths):
        image_paths = list(glob.glob('*.jpg'))

    if target_layer == 'last':
        target_layer=list(model.layers.keys())[-1]
    
    # 定义CAM方法列表
    cam_methods = [
        SmoothGradCAMpp,
        GradCAM,
        GradCAMpp,
        ScoreCAM,
        SSCAM,
        ISCAM,
        XGradCAM,
        LayerCAM,
    ]

    # 加载模型并设置为评估模式
    model.load_state_dict(torch.load(weights_path, map_location="cuda"))
    model.eval()

    for image_path in image_paths:

        print('------ ', image_path, '-------')
        
        # 读取并预处理图像
        img = read_image(image_path)
        input_tensor = normalize(
            resize(img, (224, 224)) / 255.,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

        # 创建画布
        plt.figure(figsize=(6, 6))
        
        # 显示原始图像
        plt.subplot(3, 3, 1)
        plt.imshow(to_pil_image(img))
        plt.title('Input')
        plt.axis('off')
        
        # 遍历所有CAM方法，生成并显示热力图
        for i, method in enumerate(cam_methods):
            # 确定子图位置
            plt.subplot(3, 3, i + 2)  # 注意索引偏移
            
            with method(model, target_layer=target_layer) as extractor: # , target='layer2'
        
                # 前向传播
                out = model(input_tensor.unsqueeze(0).cuda())
                if i ==0:
                    print('prediction: ', out.squeeze(0))                
                
                # 提取CAM
                if 'Fresh' in image_path:
                    activation_map = extractor(0, out)
                else:
                    activation_map = extractor(1, out)
                
                # 将热力图叠加到图像上
                result = overlay_mask(
                    to_pil_image(img),
                    to_pil_image(activation_map[0].squeeze(0), mode='F'),
                    alpha=0.5
                )
                
                # 显示热力图
                plt.imshow(result)
                plt.title(method.__name__)
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()