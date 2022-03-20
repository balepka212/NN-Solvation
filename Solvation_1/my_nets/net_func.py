import torch
import torch.nn as nn
from tqdm import tqdm


def validate(model, val_loader):
    """TODO write all descriptions"""
    total = 0
    all_MSE = 0
    loss = nn.MSELoss()
    with torch.no_grad():
        for vector, G_true in val_loader:
            vector, G_true = vector.to('cpu'), G_true.to('cpu')
            model.to('cpu')
            outputs = model(vector)
            total += G_true.size(0)
            all_MSE += loss(outputs.squeeze(), G_true.squeeze())

    return all_MSE / total


def train(model, train_loader, val_loader, loss_function, optimizer, epochs=10, device='cpu'):
    """TODO write all descriptions"""
    for epoch in range(epochs):
        hist_loss = 0
        for vector, G_true in tqdm(train_loader):  # get bacth
            vector, G_true = vector.to(device), G_true.to(device)
            model.to(device)
            outputs = model(vector)  # call forward inside
            # print(f'out: {outputs.shape}')
            # print(f'G: {G_true.shape}')

            loss = loss_function(outputs.squeeze(), G_true)  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # performs a single optimization step (parameter update).
            optimizer.zero_grad()  # sets the gradients of all optimized tensors to zero.

            hist_loss += loss.item()  # For stat only

        # writer.add_scalar("Train/Loss", hist_loss / ( len(trainloader) ), epoch)
        # train_loss_data[tag].append(hist_loss/(len(trainloader)))
        # writer.close()
        # # train_accuracy = get_accuracy(model, trainloader)
        # writer.add_scalar("Train/Acc", train_accuracy, epoch)
        # # train_acc_data[tag].append(train_accuracy)
        # writer.close()

        # test_loss = 0
        # for images, labels in tqdm(testloader): # get bacth
        #     outputs = model(images) # call forward inside

        #     loss = loss_function(outputs, labels) # calculate loss
        #     test_loss += loss.item() # For stat only
        # writer.add_scalar("Test/Loss", test_loss / ( len(testloader) ), epoch)
        # # test_loss_data[tag].append(test_loss/(len(testloader)))

        # writer.close()

        accuracy = validate(model, val_loader)
        # writer.add_scalar("Test/Acc", accuracy, epoch)
        # test_acc_data[tag].append(accuracy)
        # writer.close()

        # accuracy = validate(model, testloader)
        print(print(f'epoch {epoch} -> {accuracy}'))
    return accuracy
