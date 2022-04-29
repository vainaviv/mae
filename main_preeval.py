import torch
import util.misc as misc
import models_mae
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    i = 0
    for data_iter_step, (samples,_) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device, non_blocking=True)
        samples = torch.unsqueeze(samples[0], dim=0)
        print(samples.size())

        with torch.cuda.amp.autocast():
            _, outputs, _ = model(samples.float())
            outputs = model.unpatchify(outputs)
            output = outputs.detach().cpu().numpy()
            print(output.shape)
            img = output.squeeze().transpose(1, 2, 0)
            print(img.shape)
            img = np.clip(img, 0, 1)
            plt.imsave("./viz_pretrain/%d.png"%i, img)
            i += 1


if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.ToTensor()])
    dataset_dir = './train_sets/one_img'
    dataset_val = datasets.ImageFolder(os.path.join(dataset_dir, 'train/images'), transform=transform_train)
    # dataset_val = ConditioningDataset('train_sets/%s/train/images/dummy_class'%dataset_dir, 'train_sets/%s/train/annots'%dataset_dir, 100, 100, finetune=False)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    device = torch.device("cuda")
    if not os.path.exists("./viz_pretrain"):
        os.mkdir("./viz_pretrain")

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    model = models_mae.__dict__['mae_vit_large_patch16']()
    model.to(device)
    
    checkpoint = torch.load("./checkpoints/checkpoint-799.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=True)

    evaluate(data_loader_val, model, device)