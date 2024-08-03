import torch


def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    # print(loss, prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def FCN_loss_and_accuracy(prediction, target, f1, f2):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.L1Loss()
    cla_loss = cross_entropy_loss(prediction, target)
    fus_loss = mse_loss(f1, f2)
    # print(cla_loss, fus_loss)
    loss = cla_loss + fus_loss
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def BCE_and_accuracy(prediction, labels):
    target = 1-labels
    target = torch.stack([target, labels],dim=1)
    cross_entropy_loss = torch.nn.BCEWithLogitsLoss()
    loss = cross_entropy_loss(prediction, target.float())
    # print(loss, prediction, target)
    accuracy = (prediction.argmax(1) == target[:,1]).float().mean()
    return loss, accuracy