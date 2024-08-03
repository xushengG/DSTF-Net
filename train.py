import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm
import sys
sys.path.append('/cwang/home/gxs/mos')
from torch.utils.tensorboard import SummaryWriter
from utils.Loss import cross_entropy_loss_and_accuracy, BCE_and_accuracy
from torchvision import utils as vutils
from DvsDatas.loader import Loader
from DvsDatas.NCaltech101_pillar import NCaltech101
torch.manual_seed(1)
np.random.seed(1)


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="/Dataset/N-Caltech101/validation")
    parser.add_argument("--training_dataset", default="/Dataset/N-Caltech101/training")

    # logging options
    parser.add_argument("--log_dir", default="Log")
    parser.add_argument("--model_save", default="Log")
    parser.add_argument("--checkpoint", default="Log/model_best.pth")

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=12)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)

    flags = parser.parse_args()

    if os.path.exists(flags.log_dir) == False:
            os.makedirs(flags.log_dir, exist_ok=True)
    if os.path.exists(flags.model_save) == False:
            os.makedirs(flags.model_save, exist_ok=True)

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
          f"----------------------------")
    
    return flags

if __name__ == '__main__':
    flags = FLAGS()
    # device = torch.device(flags.device if torch.cuda.is_available() else "cpu") 

    # datasets, add augmentation to training set
    training_dataset = NCaltech101(root=flags.training_dataset, augmentation=True)
    validation_dataset = NCaltech101(root=flags.validation_dataset, augmentation=False)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=flags.batch_size,
                                             num_workers=flags.num_workers, pin_memory=True,
                                             shuffle=True,
                                             drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=flags.batch_size,
                                             num_workers=flags.num_workers, pin_memory=True,
                                             drop_last=True)

    from Networks.DSTF_Net import get_model
    model = get_model()

    # ckpt = torch.load(flags.checkpoint)
    # model.load_state_dict(ckpt["state_dict"])
    
    model = model.to(flags.device)
    criterion = cross_entropy_loss_and_accuracy
    # criterion = BCE_and_accuracy
    # criterion.to(flags.device)

    # optimizer and lr scheduler
    # initial_lr = 0.1
    # final_lr = 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.paralmeters(), lr=1e-4, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=flags.num_epochs, eta_min=final_lr)

    writer = SummaryWriter(flags.log_dir)

    iteration = 0
    min_validation_loss = 1000

    for i in range(flags.num_epochs):
        if i != 0 and i % flags.save_every_n_epochs == 0:
            sum_accuracy = 0
            sum_loss = 0
            model = model.eval()

            print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
            for fus, events, labels in tqdm.tqdm(validation_loader):
                fus = fus.to(flags.device)
                events = events.to(flags.device)
                labels = labels.to(flags.device)
                with torch.no_grad():
                    # print(fus.shape, events.shape)
                    pred_labels = model(fus, events)
                    loss, accuracy = criterion(pred_labels, labels.long())

                sum_accuracy += accuracy
                sum_loss += loss

            validation_loss = sum_loss.item() / len(validation_loader)
            validation_accuracy = sum_accuracy.item() / len(validation_loader)

            writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
            writer.add_scalar("validation/loss", validation_loss, iteration)

            print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                state_dict = model.state_dict()

                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_validation_loss,
                    "iteration": iteration
                }, os.path.join(flags.model_save,"model_best.pth"))
                print("New best at loss", validation_loss)
            
            # if i % flags.save_every_n_epochs == 0:
            #     state_dict = model.state_dict()
            #     torch.save({
            #         "state_dict": state_dict,
            #         "min_val_loss": min_validation_loss,
            #         "iteration": iteration
            #     },  os.path.join(flags.model_save, "checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss)))

        sum_accuracy = 0
        sum_loss = 0
        min_train_loss = 1000
        model = model.train()
        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
        for fus, events, labels in tqdm.tqdm(training_loader):
            optimizer.zero_grad()
            fus = fus.to(flags.device)
            events = events.to(flags.device)
            labels = labels.to(flags.device)
            pred_labels = model(fus, events)
            
            loss, accuracy = criterion(pred_labels, labels.long())

            loss.backward()
            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1

        if i % 10 == 9:
            lr_scheduler.step()

        training_loss = sum_loss.item() / len(training_loader)
        training_accuracy = sum_accuracy.item() / len(training_loader)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)
        if training_loss < min_train_loss:
                min_train_loss = training_loss
                state_dict = model.state_dict()

                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_train_loss,
                    "iteration": iteration
                }, os.path.join(flags.model_save,"trainloss_best.pth"))
                # print("New best at loss", training_loss)
