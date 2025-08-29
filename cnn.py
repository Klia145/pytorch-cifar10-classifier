import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os 
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# --- 新增这两行 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

def Controll_argparse():
    parser=argparse.ArgumentParser(description='CIFAR-10 分类模型训练与评估工具')
    parser.add_argument('--model',type=str,default='MiniVGG',choices=['CNN','MiniVGG']
                        ,help='选择要使用的模模型:CNN或者MiniVGG')
    parser.add_argument('--mode',type=str,default='train',choices=['train','eval'],
                        help='选择运行模式:train(训练)或者eval(评估)')
    parser.add_argument('--epochs',type=int,default=10,help='训练轮数')
    parser.add_argument('--version',type=int,default=0,help='模型版本')
    args=parser.parse_args()
    print(f"--- 模式: {args.mode} | 模型: {args.model} ---")
    return args
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,16,4)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,32,5)
        self.bn2=nn.BatchNorm2d(32)
        self.maxpool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        self.dropout=nn.Dropout(p=0.2)
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)
        
        x=self.maxpool(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc3(x)
        
        return x

class MiniVGG(nn.Module):
    def __init__(self):
        super(MiniVGG, self).__init__()
        
        # 块 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # 块 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 块 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 经过3次池化，32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 块 1
        x = F.relu(self.conv1_1(x))
        x = self.pool(F.relu(self.conv1_2(x)))
        
        # 块 2
        x = F.relu(self.conv2_1(x))
        x = self.pool(F.relu(self.conv2_2(x)))
        
        # 块 3
        x = F.relu(self.conv3_1(x))
        x = self.pool(F.relu(self.conv3_2(x)))
        
        # 压平并分类
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# 这是一个更通用的版本
def get_next_version_number(model_dir, model_prefix):
    """
    检查指定文件夹，根据模型名前缀返回下一个版本号。
    例如: get_next_version_number('models/MiniVGG', 'minivgg')
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) # 如果文件夹不存在，就创建它
        return 1
        
    existing_files = os.listdir(model_dir)
    if not existing_files:
        return 1
        
    numbers = []
    for file in existing_files:
        # 确保文件符合 "prefix_number.pth" 的格式
        if file.startswith(model_prefix) and file.endswith('.pth'):
            try:
                # 尝试从文件名中解析出数字
                number_str = file.split('_')[-1].split('.')[0]
                numbers.append(int(number_str))
            except (ValueError, IndexError):
                # 如果文件名格式不符，就跳过
                continue
                
    return max(numbers) + 1 if numbers else 1
        
def train_model(model,train_loader,device,epochs=2,optimizer=None,criterion=None,scheduler=None):
    loss_history=[
    ]
    accuracy_history=[]
    if device is None:
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if optimizer is None:
        optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    if criterion is None:
        criterion=nn.CrossEntropyLoss()
    if model is None:
        model=CNN()
    model_name=model.__class__.__name__
    print(f'开始训练{model_name}模型')
    for epoch in range(epochs):
        running_loss=0
        for i,data in enumerate(train_loader,0):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            
            optimizer.zero_grad()
            
            outputs=model(inputs)
            
            loss=criterion(outputs,labels)
            
            loss.backward()
            
            optimizer.step()
            
            running_loss+=loss.item()
            
            if i%200 ==199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        epoch_loss=running_loss/len(train_loader)
        loss_history.append(epoch_loss)
        epoch_accuracy=test_accuracy(model,train_loader,device)
        accuracy_history.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{epochs} - 训练损失: {epoch_loss:.4f}, 测试准确率: {epoch_accuracy:.2f}%")

        
        if(scheduler is not None):
            scheduler.step()
    print('Finished Training')
    return loss_history,accuracy_history

def view_data(loss_history,accuracy_history,args,version_number):
    print("正在绘制学习曲线")
    fig,ax1=plt.subplots()
    
    color='tab:red'
    
    ax1.set_xlabel('Epochs')
    
    ax1.set_ylabel('Loss',color=color)
    ax1.plot(loss_history,color=color,label='Loss')
    ax1.tick_params(axis='y',labelcolor=color)
    
    ax2=ax1.twinx()
    color='tab:blue'
    ax2.set_ylabel('测试准确率 (Accuracy %)', color=color)
    ax2.plot(accuracy_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout() # 自动调整布局
    plt.title(f'{args.model} 学习曲线')
    curve_filename=f"{args.model.lower()}_curve_{version_number:03d}.png"
    curves_dir=os.path.join("curve",args.model)
    os.makedirs(curves_dir,exist_ok=True)
    curve_save_path=os.path.join(curves_dir,curve_filename)
    plt.savefig(curve_save_path)
    print(f"学习曲线保持到 {curve_save_path}")
    
    plt.show()
    
def test_accuracy(model,test_loader,device):
    if device is None:
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images.to(device))
            _,predicted=torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    accuracy=100*correct/total
    return accuracy
def load_trained_model(model, model_path, device):
    print(f"正在从 {model_path} 加载模型...")
    # 1. 从文件加载权重字典
    state_dict = torch.load(model_path, map_location=device)
    # 2. 将权重加载到模型中（这行代码在任何情况下都必须执行）
    model.load_state_dict(state_dict)
    # 3. 将模型移动到指定设备
    model.to(device)
    # 4. 切换到评估模式
    model.eval()
    print("模型加载完毕。")
    return model
        

def main():
    args=Controll_argparse()
    transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
    current_working_directory = os.getcwd()

    print(f"脚本当前的'立足点'(工作目录)是: {current_working_directory}")
    data_folder_path = 'D:\Program\DeepLearning\CNN\data'
    trainset=torchvision.datasets.CIFAR10(root=data_folder_path,train=True,
                                    download=True,transform=transform)

    testset=torchvision.datasets.CIFAR10(root=data_folder_path,train=False,
                                    download=True,transform=transform)
    num_cpu_workers = 0 

    print(f"--- 使用 {num_cpu_workers} 个CPU核心进行数据加载 ---")

    trainloader=DataLoader(trainset,batch_size=64,shuffle=True, num_workers=num_cpu_workers)
    testloader=DataLoader(testset,batch_size=64,shuffle=False, num_workers=num_cpu_workers)
    classes=('plane','car','bird','cat','dear','dog','frog','horse','ship','truck')

    dataiter=iter(trainloader)
    images,labels=next(dataiter)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model=='CNN':
        net=CNN().to(device)
    elif args.model=='MiniVGG':
        net=MiniVGG().to(device)
    else:
        raise ValueError(f'未知的模型类型: {args.model}')
    model_dir = f"model/{args.model}"
    model_save_path=f'model/{args.model}/{args.model.lower()}.pth'
    
    
    MODELS_BASE_DIR=os.path.join(SCRIPT_DIR, 'model')
    model_specific_dir = os.path.join(MODELS_BASE_DIR, args.model)

    if args.mode == 'train':
        print(f"开始新的训练...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001) # Adam通常效果更好
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # 10个epoch衰减一次
        
        loss_history,accuracy_history=train_model(net, trainloader, device, epochs=args.epochs, optimizer=optimizer, criterion=criterion, scheduler=scheduler)
        
        version_number=get_next_version_number(model_specific_dir,args.model.lower())
        model_filename=f"{args.model.lower()}_{version_number:03d}.pth"
        model_save_path=os.path.join(model_specific_dir,model_filename)
        torch.save(net.state_dict(), model_save_path)
        view_data(loss_history,accuracy_history,args,version_number)
        
        print(f"保存模型到: {model_save_path}")

    elif args.mode == 'eval':

        # 在评估模式下，我们需要找到【最新】的一个模型文件来加载
        version_number=get_next_version_number(model_specific_dir,args.model.lower())
        if args.version==0:
            latest_version=get_next_version_number(model_specific_dir,args.model.lower())-1
            if latest_version<1:
                print(f"错误：未找到模型文件: {model_load_path}")
                return
            version_number=latest_version
        else:
            version_number=args.version
            print(f"使用版本号为 {version_number} 的模型文件")
        model_filename = f"{args.model.lower()}_{version_number:03d}.pth"
        model_load_path = os.path.join(model_dir, model_filename)

        if os.path.exists(model_load_path):
            net = load_trained_model(net, model_load_path, device)
        else:
            print(f"错误：未找到模型文件: {model_load_path}")
            return
    
    accuracy_rest=test_accuracy(net,testloader,device)
    print(f'模型在10000张测试图片上的准确率是: {accuracy_rest:.2f} %')
    images_for_model=[]
    images_for_plot=[]
    actual_labels=[]
    for i in range(4):
        image,label=testset[i]
        image_for_plot=image*0.5+0.5
        image_for_plot=image_for_plot.permute(1,2,0)
        images_for_model.append(image)
        images_for_plot.append(image_for_plot)
        actual_labels.append(label)
    batch_tensor=torch.stack(images_for_model).to(device)
    with torch.no_grad():
        outputs = net(batch_tensor)
        _, predicted_indices = torch.max(outputs, 1)
    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)  
        plt.imshow(images_for_plot[i]) 
        actual_name = classes[actual_labels[i]]
        predicted_name = classes[predicted_indices[i].item()] 
        plt.title(f"预测: {predicted_name}\n真实: {actual_name}")
        plt.axis('off') 
    plt.tight_layout()
    plt.show()

    
def preidct_colors(model,color_rgb):
    color_rgb=torch.tensor(color_rgb,dtype=torch.float32).unsqueeze(0)/ 255.0
    
if __name__ == '__main__':
    main()