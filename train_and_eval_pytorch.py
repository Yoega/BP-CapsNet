
import os
import argparse
import sys
import time
import random
import numpy as np
from tqdm import trange
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# set random seed
setup_seed(3407)
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from tensorboardX import SummaryWriter
from collections import OrderedDict
# sys.path.append('/swin/WingaLey/Inner_product/base/RAW')
sys.path.append('/home/zyli/exper/CapsNet/capsnet/Inner_product/base/RAW')
from RAW import capsnet as caps
from models import ResNet18, ResNet50

import medmnist
from medmnist import INFO
from denoise import GaussianBlur, MeanFilter
class_n = 9
channel_n = 3
# root = '/home/swin/WingaLey/Inner_product/data'
root = '/home/zyli/exper/CapsNet/capsnet/Inner_product/data'



def main(data_flag, output_root, num_epochs, gpu_ids, batch_size,
         download, model_flag, resize, as_rgb, model_path, run):
    lr = 1e-4
    gamma = 0.96

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')


    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    print(output_root)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop(28, padding=2),
             transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, root=root)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, root=root)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, root=root)


    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')


    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model =  resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    # add capsule network
    elif model_flag == 'capsnet':
        model = caps.CapsNet()
    else:
        raise NotImplementedError

    model = model.to(device)

    train_evaluator = medmnist.Evaluator(data_flag, 'train', root=root)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', root=root)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', root=root)

    # if task == "multi-label, binary-class":
    #     criterion = nn.BCEWithLogitsLoss()
    # else:
    #     criterion = nn.CrossEntropyLoss()
    criterion = caps.CapsuleLoss().to(device)


    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)

    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = model

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer)



        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)

        scheduler.step()

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = model
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)

    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)

    writer.close()


def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        """
        outputs = model(inputs.to(device))
        """

        # # capsule network model
        # # # SVD module
        # num = 12
        # inputs = torch.reshape(inputs, [-1, 28*channel_n, 28])
        # U, sigma, VT = torch.linalg.svd(inputs)
        # inputs = torch.matmul((torch.matmul(U[:, :, 0:num], torch.diag_embed(sigma)[:, 0:num, 0:num])), VT[:, 0:num, :])
        # inputs = torch.reshape(inputs, [-1, channel_n, 28, 28]).to(device)

        # mean value filter
        mean_filter = MeanFilter(3)
        inputs = mean_filter(inputs).to(device)

        # # # Guassian Blur
        # inputs = GaussianBlur(inputs, 3).to(device)

        # inputs = inputs.to(device)
        targets = torch.squeeze(targets, dim=1)
        targets = torch.eye(class_n).index_select(dim=0, index=targets).to(device)
        outputs, reconstruction = model(inputs)

        """
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
        """
        # Compute loss & accuracy
        # --> traverse to tensor
        loss = criterion(inputs, targets, outputs, reconstruction)
        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            """
            outputs = model(inputs.to(device))
            """
            # outputs and reconstruction of testinputs.to(device)
            inputs = inputs.to(device)
            outputs, reconstruction = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                """
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                """
                targets = torch.squeeze(targets, dim=1)
                targets = torch.eye(class_n).index_select(dim=0, index=targets).to(device)
                loss = criterion(inputs, targets, outputs, reconstruction)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)


        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)
        print(test_loss, auc, acc)
        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./results',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='2',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    # parser.add_argument('--download',
    #                     action="store_true")
    # parser.add_argument('--resize',
    #                     help='resize images of size 28x28 to 224x224',
    #                     action="store_false")
    # parser.add_argument('--as_rgb',
    #                     help='convert the grayscale image to RGB',
    #                     action='store_true')
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='capsnet',
                        help='choose backbone from resnet18, resnet50, capsnet',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)

    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = True
    model_flag = args.model_flag
    resize = False
    as_rgb = True
    model_path = args.model_path
    run = args.run

    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run)
