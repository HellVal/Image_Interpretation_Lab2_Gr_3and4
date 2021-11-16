# -*- coding: utf-8 -*-
import os
import time
import copy
# import h5py
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torchvision.models.segmentation.fcn import FCNHead, FCN
import server_train_data_loader_yatao

import server_regression_model
#import torch.optim.lr_scheduler
#import torch.optim.lr_scheduler.ReduceLROnPlateau as lro
from random import randint

def train_model(model, dataloaders, criterion, optimizer, num_epochs, log_path):
    begin_time = time.time()
    val_acc_history = ['valid']
    tri_acc_history = ['train']
    log_list = []
    time_list = []
    memo_list = []


    best_model_ets = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    # For each epoch, tran and evaluate the model
    for epoch in range(1, num_epochs+1):
        print("Epoch {}/{}---".format(epoch, num_epochs))
        log_list.append("Epoch {}/{}---".format(epoch, num_epochs))
        
        
        train_counter = 0
        for phase in ['train', 'valid']:
            counter = 0
            if phase == "train":
                model.train()
                print("Train with {} samples".format(len(dataloaders[phase])))
                log_list.append("Train with {} samples".format(len(dataloaders[phase])))
            else:
                model.eval()
                print("Eval with {} samples".format(len(dataloaders[phase])))
                log_list.append("Eval with {} samples".format(len(dataloaders[phase])))

            running_loss = 0.0

            # Iterate
            flag = 1
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                
                optimizer.zero_grad()

                # forward, track history in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)  
                    outputs_masked = torch.squeeze(outputs)
                    
                    #generte mask to filter out all pixels with label -1
                    mask = labels.ge(0.)
                    mask = torch.squeeze(mask)
                    labels_masked = torch.squeeze(labels)
                    #print(mask.sum())
                    if mask.sum() == 0:
                        counter = counter+1
                        print('Hello')
                        continue
                    
                    loss = criterion(outputs_masked[mask], labels_masked[mask])
                    
                    ## visualization of each 50th image
                    if train_counter%50 == 0:
                        s_inputs = inputs.shape
                        random_number = randint(0,s_inputs[0]-1)
                        plot_inputs = inputs.cpu().detach().numpy()
                        plot_inputs = np.transpose(plot_inputs,(0,2,3,1))
                        plot_labels =labels.cpu().detach().numpy()
                        plot_outputs = outputs.cpu().detach().numpy()
                        
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                        fig.suptitle('Actuall loss: '+str(loss.item()))
                        ax1.imshow(plot_inputs[random_number,:,:,:3])
                        ax1.title.set_text('input data RGB')
                        ax2.imshow(plot_labels[random_number,:,:])
                        ax2.title.set_text('labels')
                        
                        
                        ax3.imshow(plot_outputs[random_number,0,:,:])
                        ax3.title.set_text('output')
                        fig.savefig('./output/'+str(epoch)+'_'+str(train_counter)+'.png')
                        plt.close()


                    # backward and optimize in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    time_list.append("{},{},{}".format(epoch, phase, time.time() - begin_time))
                    memo_list.append("{},{},{}".format(epoch, phase, psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
                    flag += 1
                    if flag % 100 == 0:
                        print("data-{}".format(flag))

                # statistics
                running_loss += loss.item()
                train_counter = train_counter+1
        


            # print("len(dataloaders[phase].dataset): {}".format(len(dataloaders[phase].dataset)))
            epoch_loss = running_loss / (len(dataloaders[phase].dataset)-counter)
            # epoch_loss = running_loss / len(dataloaders[phase])
            print('{} loss: {:.4f}'.format(phase, epoch_loss))
            log_list.append('{} loss: {:.4f}'.format(phase, epoch_loss))

            # copy the model when it performs best
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_ets = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_loss)
            if phase == 'train':
                tri_acc_history.append(epoch_loss)
                
    

    time_used = time.time() - begin_time
    print('Training time: {}'.format(time_used))
    print('Best validation acc: {}'.format(best_loss))
    log_list.append('Training time: {}'.format(time_used))
    log_list.append('Best validation acc: {}'.format(best_loss))
    df_log = pd.DataFrame(log_list)
    df_log.to_csv(log_path+"_state.txt", index=False)
    df_time = pd.DataFrame(time_list)
    df_time.to_csv(log_path+"_time.csv", index=False)
    df_memo = pd.DataFrame(memo_list)
    df_memo.to_csv(log_path+"_memo.csv", index=False)
    df_loss = pd.DataFrame([tri_acc_history, val_acc_history])
    df_loss.to_csv(log_path+"_loss.csv", index=False)

    model.load_state_dict(best_model_ets)
    return model, val_acc_history


def viz(module, _input, _output):
    x = _input[0][0]
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i])
    plt.show()


if __name__ == '__main__':

    root = 'P:/pf/pfstud/II_Group3/final_data/'

    # dataset and some basic variables
    dataTraining = root + '/train_data.hdf5'
    #dataTraining = root+'/black_image.hdf5'
    dataset = server_train_data_loader_yatao.ImageLoader(windowsize=256, test=False, datafile=dataTraining)
    train_size = int(len(dataset) * 0.75)
    valid_size = len(dataset) - train_size



    num_output = 1  # regression task
    num_epochs = 30
    num_batches = 4
    # generate train loader and validation loader
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=num_batches,
                                               num_workers=6,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=num_batches,
                                               num_workers=6,
                                               shuffle=True)
    dataloaders_dict = {'train': train_loader, 'valid': valid_loader}
    print("Train size: {}".format(len(train_loader)))
    print("Valid size: {}".format(len(valid_loader)))

    # determine whether gpu will be used in the training process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # select segmentation model
    model_ft = server_regression_model.UNet2().to(device)



    # create the optimizer
    model_ft = model_ft.to(device)
    # params_to_update = model_ft.parameters()
    print("Parameters need to update:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:  # only the fc layer
            params_to_update.append(param)
            print("\t", name)
    optimizer_ft = optim.SGD(params_to_update, lr=0.000001, momentum=0.9)
    

    # setup the loss function
    criterion = nn.MSELoss()
    

    # train and validation
    log_path = "UNET"
    print('before training, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    model_ft, hist_acc = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs, log_path)
    print('after training, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    torch.save(model_ft, "model_UNET_best")
    torch.save(model_ft.state_dict(), "model_UNET_state_dict_best")
    print("hist_acc:")
    print(hist_acc)
