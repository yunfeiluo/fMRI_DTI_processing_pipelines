import torch
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

import pickle
import matplotlib.pyplot as plt
import gc
import nibabel as nib

from ResNet18AE import *
from ResNet34AE import *

def train_model(
    samples, 
    labels, 
    epochs, 
    generator,
    discriminator, 
    reconstruction_loss_func, 
    detect_loss_func,
    g_optimizer,
    d_optimizer, 
    g_scheduler,
    d_scheduler,
    batch_size,
    lam_1,
    lam_2,
    lam_3
):
    train_rec_loss = list()
    val_rec_loss = list()
    train_dis_loss = list()
    val_dis_loss = list()
    
    train_num = int(len(samples) * 0.8)
    train_samples = samples[:train_num]
    # train_labels = labels[:train_num]
    val_samples = samples[train_num:]
    # val_labels = labels[train_num:]

    print('data shape:', samples.shape)
    print('num epochs:', epochs)
    print('num data:', len(samples))
    print('num train:', train_num)
    print('batch size:', batch_size)

    for epoch in range(epochs):
        # train discriminator
        print('Train Discriminator...')
        discriminator.train()
        generator.eval()
        discriminator_train_loss = 0
        discriminator_val_loss = 0
        i = 0
        while i < len(samples):
            curr_samples = None
            if i < train_num:
                curr_samples = train_samples[i:i+batch_size]
            else:
                discriminator.eval()
                curr_samples = samples[i:i+batch_size]

            # feed real image for forward
            classify_real_out = discriminator(curr_samples)
            classify_real_loss = detect_loss_func(classify_real_out, torch.ones(len(curr_samples), dtype=torch.long))

            # feed fake image from generator
            encode_out, decode_out = generator(curr_samples)
            classify_fake_out = discriminator(decode_out)
            classify_fake_loss = detect_loss_func(classify_fake_out, torch.zeros(len(curr_samples), dtype=torch.long))

            total_loss = classify_real_loss + classify_fake_loss

            # backward and update parameters
            if i < train_num:
                discriminator_train_loss += total_loss.detach().item() * len(curr_samples)
                discriminator.zero_grad()
                total_loss.backward()
                d_optimizer.step()
            else:
                discriminator_val_loss += total_loss.detach().item() * len(curr_samples)
                discriminator.zero_grad()
                total_loss.backward()
                discriminator.zero_grad()
                # d_optimizer.step()                

            # update i
            i += len(curr_samples)
            # gc.collect()
        
        # train generator
        print('Train Generator...')
        generator.train()
        discriminator.eval()
        generator_train_loss = 0
        generator_val_loss = 0
        i = 0
        while i < len(samples):
            curr_samples = None
            if i < train_num:
                curr_samples = train_samples[i:i+batch_size]
            else:
                generator.eval()
                curr_samples = samples[i:i+batch_size]

            # forward            
            encode_out, decode_out = generator(curr_samples)
            reconstruction_loss = reconstruction_loss_func(decode_out, curr_samples)
            reconstruct_latent, _ = generator(decode_out)
            latent_reconstruction_loss = reconstruction_loss_func(reconstruct_latent, encode_out)
            classify_out = discriminator(decode_out)
            classify_real_loss = detect_loss_func(classify_out, torch.ones(len(curr_samples), dtype=torch.long))
    
            total_loss = (lam_1 * reconstruction_loss + lam_2 * latent_reconstruction_loss + lam_3 * classify_real_loss)

            # backward and update parameters
            if i < train_num:
                generator_train_loss += total_loss.detach().item() * len(curr_samples)
                generator.zero_grad()
                total_loss.backward()
                g_optimizer.step()
            else:
                generator_val_loss += total_loss.detach().item() * len(curr_samples)
                generator.zero_grad()
                total_loss.backward()
                generator.zero_grad()
                # g_optimizer.step()

            # update i
            i += batch_size
            # gc.collect()

        # update
        train_rec_loss.append(generator_train_loss)
        val_rec_loss.append(generator_val_loss)
        train_dis_loss.append(discriminator_train_loss)
        val_dis_loss.append(discriminator_val_loss)

        g_scheduler.step()
        d_scheduler.step()

        # verbose
        print("================================================")
        print('epoch', epoch)
        print('G train loss', generator_train_loss / len(train_samples))
        print('D train loss', discriminator_train_loss / len(train_samples))
        print("================================================")
        print('G val loss', generator_val_loss / len(val_samples))        
        print('D val loss', discriminator_val_loss / len(val_samples))
        print("================================================")
        print(' ')
    return generator, discriminator, train_rec_loss, val_rec_loss, train_dis_loss, val_dis_loss

if __name__ == '__main__':
    # declare device
    device = torch.device('cpu')

    # fetch data
    imgs = list()
    affines = list()
    headers = list()
    problem_sub = list()
    for i in range(1, 27):
        try:
            img = nib.load('healthy_data_mni/{}/standardized_struct_file/wrsub{}_brain.nii'.format(i, i))
            affines.append(img.affine)
            headers.append(img.header)
            img = img.get_fdata()
            assert(img.shape == (182, 218, 182))
            img = np.nan_to_num(img.reshape(1, 182, 218, 182))
            # assert(img.shape == (91, 109, 91))
            # img = np.nan_to_num(img.reshape(1, 91, 109, 91))
            imgs.append(img)
        except:
            problem_sub.append(i)
            continue
    print('problem sub', problem_sub)
    imgs = np.array(imgs)
    
    # hyperparameters
    epochs = 10
    batch_size = 1
    global_lr = 1e-3
    reg = 1e-8 # L2 penalty
    g_begin_channel = 64

    # for discriminator
    d_lr = 1e-3
    d_reg = 1e-8
    d_begin_channel = 64

    # for loss
    lam_1 = 1.0 # coef of reconstruction loss
    lam_2 = 5e-1 # regularize reconstruct latent (might remove this loss with modification on structure)
    lam_3 = 1.0 # coef of loss from discriminator

    # construct model
    data = torch.Tensor(imgs).to(device)
    generator = Res18Autoencoder(begin_channel=g_begin_channel).to(device)
    # generator = None
    # with open('output/saved_models/generator', 'rb') as f:
    #     generator = pickle.load(f)
    reconstruction_loss_func = nn.MSELoss()
    # print('generator', generator)

    discriminator = Res18Encoder(begin_channel=g_begin_channel, num_classes=2).to(device)
    # discriminator = None
    # with open('output/saved_models/discriminator', 'rb') as f:
    #     discriminator = pickle.load(f)
    detect_loss_func = nn.CrossEntropyLoss()
    # print('discriminator', discriminator)

    # define optimizer and scheduler for lr decay
    g_optimizer = optim.Adam(
        [
            {'params': generator.encoder.parameters()},
            {'params': generator.decoder.parameters()}
        ],
        lr=global_lr,
        weight_decay=reg
    )

    d_optimizer = optim.Adam(
        [
            {'params': discriminator.parameters()}
        ],
        lr=d_lr,
        weight_decay=d_reg
    )

    g_scheduler = StepLR(g_optimizer, step_size=1, gamma=0.95)
    d_scheduler = StepLR(d_optimizer, step_size=1, gamma=0.95)

    # train model
    generator, discriminator, train_rec_loss, val_rec_loss, train_dis_loss, val_dis_loss = train_model(
        samples=data, 
        labels=data, 
        epochs=epochs, 
        generator=generator,
        discriminator=discriminator, 
        reconstruction_loss_func=reconstruction_loss_func, 
        detect_loss_func = detect_loss_func,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer, 
        g_scheduler=g_scheduler,
        d_scheduler=d_scheduler,
        batch_size=batch_size,
        lam_1=lam_1,
        lam_2=lam_2,
        lam_3=lam_3
    )

    # save file example code
    reconstructed_imgs = list()
    residual_imgs = list()
    for i in range(len(imgs)):
        _, reconstructed_img = generator(torch.Tensor([imgs[i]]).to(device))
        reconstructed_img = reconstructed_img.detach().numpy().astype('float64')[0]
        reconstructed_imgs.append(reconstructed_img)
        residual_imgs.append(np.abs(imgs[i] - reconstructed_img))
    reconstructed_imgs = np.array(reconstructed_imgs)
    residual_imgs = np.array(residual_imgs)

    # visualize training loss
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(train_rec_loss,label="G")
    plt.plot(train_dis_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Validation")
    plt.plot(val_rec_loss,label="G")
    plt.plot(val_dis_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # save to file
    save = input('do you want to save the results?(y/n)')
    if save == 'n':
        exit()

    for i in range(len(reconstructed_imgs)):
        test_img = nib.Nifti1Image(reconstructed_imgs[i][0], affines[i], headers[i])
        test_img.to_filename('output/reconstruct_output/sub{}_reconstruct.nii'.format(i))

        residual_img = nib.Nifti1Image(residual_imgs[i][0], affines[i], headers[i])
        residual_img.to_filename('output/residual_output/sub{}_residual.nii'.format(i))
    
    with open('output/saved_models/generator', 'wb') as f:
        pickle.dump(generator, f)
    
    with open('output/saved_models/discriminator', 'wb') as f:
        pickle.dump(discriminator, f)
