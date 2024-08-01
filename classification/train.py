import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transnext import transnext_micro as creatModel  # 假设你有一个creatModel函数来创建模型
import time
import sys
from thop import profile

# 定义优化器
optimizers = {
    'Adam': optim.Adam
    # 'SGD': optim.SGD,
    # 'RMSprop': optim.RMSprop,
}

# 定义损失函数
loss_functions = {
    'CrossEntropyLoss': nn.CrossEntropyLoss()
    # 'MSELoss': nn.MSELoss(),
}

# 用于设置数据加载器的随机种子
def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)

# 打印模型参数数量
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    return total_params

# 打印模型的 FLOPs
def print_model_flops(model, input_size):
    input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
    model.to(next(model.parameters()).device)  # 确保模型也在同一设备上
    flops, params = profile(model, inputs=(input_tensor,))
    print(f'Total FLOPs: {flops}')
    return flops

# 主函数
def main():
    global best_loss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "SCUT-FBP5500")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    lr = 0.0001
    note = 'TransNeXt-micro-warmup25'
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw,
                                               worker_init_fn=worker_init_fn)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=nw, worker_init_fn=worker_init_fn)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model_weight_path = r'./transnext_micro_224_1k.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    def load_model():
        net = creatModel(num_classes=1000)
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        in_channel = net.head.in_features
        net.head = nn.Linear(in_channel, 5)
        return net

    model = load_model()
    print_model_params(model)
    print_model_flops(model, input_size=(batch_size, 3, 224, 224))

    epochs = 20

    results = []

    # 遍历所有优化器和损失函数的组合
    for optim_name, optim_func in optimizers.items():
        for loss_name, loss_function in loss_functions.items():
            print(f"Training with optimizer: {optim_name}, loss function: {loss_name}")

            net = load_model()
            net.to(device)

            optimizer = optim_func(net.parameters(), lr=lr)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)

            # 定义warmup调度器
            warmup_epochs = 3
            warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_epochs)

            accumulation_steps = 1  # 进行几次梯度累积后再更新权重
            if not os.path.exists('./weights'):
                os.makedirs('./weights')

            best_acc = 0.0
            best_loss = 3.0

            total_filename = "{}-lr{}-bs{}-{}-{}-{}".format(epochs, lr, batch_size, note, optim_name, loss_name)
            log_dir = "./runs/{}".format(total_filename)  # 自定义日志目录名称
            tb_writer = SummaryWriter(log_dir=log_dir)
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)  # 这里用批量大小调整
            tb_writer.add_graph(net, dummy_input)

            train_start_time = time.perf_counter()

            for epoch in range(epochs):
                net.train()
                running_loss = 0.0
                train_bar = tqdm(train_loader, file=sys.stdout)
                correct_train = 0
                optimizer.zero_grad()  # 在每个epoch开始前清空梯度

                for step, data in enumerate(train_bar):
                    images, labels = data
                    original_labels = labels.clone()  # 保留原始标签
                    images, labels = images.to(device), labels.to(device)
                    if loss_name == 'MSELoss':
                        labels = F.one_hot(labels, num_classes=5).float()
                    logits = net(images)
                    loss = loss_function(logits, labels)
                    loss.backward()

                    if (step + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    running_loss += loss.item()
                    predict_y = torch.max(logits, dim=1)[1]
                    correct_train += torch.eq(predict_y, labels).sum().item()  # 使用labels而不是original_labels
                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

                if (step + 1) % accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_acc = correct_train / train_num

                net.eval()
                acc = 0.0
                running_val_loss = 0.0
                with torch.no_grad():
                    val_bar = tqdm(validate_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        original_val_labels = val_labels.clone()  # 保留原始标签
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        if loss_name == 'MSELoss':
                            val_labels = F.one_hot(val_labels, num_classes=5).float()
                        outputs = net(val_images)
                        loss = loss_function(outputs, val_labels)
                        running_val_loss += loss.item()
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels).sum().item()  # 使用val_labels而不是original_val_labels
                        val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

                val_acc = acc / val_num
                running_loss /= len(train_loader)
                running_val_loss /= len(validate_loader)

                tb_writer.add_scalar('train_loss', running_loss, epoch)
                tb_writer.add_scalar('train_acc', train_acc, epoch)
                tb_writer.add_scalar('val_loss', running_val_loss, epoch)
                tb_writer.add_scalar('val_acc', val_acc, epoch)
                tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

                print('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' %
                      (epoch + 1, running_loss, running_val_loss, train_acc, val_acc))

                if val_acc > best_acc:
                    best_acc = val_acc
                    model_filename = "./weights/model-{}-{}-{}-{}-{}-{}-acc.pth".format(epochs, lr, batch_size, note, optim_name, loss_name)
                    torch.save(net.state_dict(), model_filename)
                    print(f"Epoch {epoch}: 保存新最佳模型 {model_filename}，验证准确率: {val_acc:.4f}")

                if running_val_loss < best_loss:
                    best_loss = running_val_loss
                    model_filename = "./weights/model-{}-{}-{}-{}-{}-{}-loss.pth".format(epochs, lr, batch_size, note, optim_name, loss_name)
                    torch.save(net.state_dict(), model_filename)
                    print(f"Epoch {epoch}: 保存新最佳模型 {model_filename}，验证损失: {running_val_loss:.4f}")

                scheduler.step()

            train_end_time = time.perf_counter()
            total_train_time = train_end_time - train_start_time
            tb_writer.add_text('Total Training Time', f"Training time: {total_train_time:.2f} seconds")

            results.append({
                'optimizer': optim_name,
                'loss_function': loss_name,
                'best_val_acc': best_acc,
                'best_val_loss': best_loss,
                'total_train_time': total_train_time,
            })

            with open('results.json', 'a') as f:
                json.dump(results, f, indent=4)

            tb_writer.close()

if __name__ == '__main__':
    main()
