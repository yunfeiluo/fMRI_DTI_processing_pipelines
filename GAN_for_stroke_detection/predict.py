import pickle
import numpy as np
import nibabel as nib
import sys
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # declare subjects names
    # subjects = ['ZF36621', 'ZF44516']
    subjects = ['example']

    # load model
    model = None
    with open('output/saved_models/generator', 'rb') as f:
        model = pickle.load(f)
    
    # fetch test data
    imgs = list()
    affines = list()
    headers = list()
    problem_sub = list()
    for sub in subjects:
        try:
            img = nib.load('test_data/{}/wrstruct_brain.nii'.format(sub))
            affines.append(img.affine)
            headers.append(img.header)
            img = img.get_fdata()
            assert(img.shape == (182, 218, 182))
            img = np.nan_to_num(img.reshape(1, 182, 218, 182))
            # assert(img.shape == (91, 109, 91))
            # img = np.nan_to_num(img.reshape(1, 91, 109, 91))
            imgs.append(img)
        except:
            problem_sub.append(sub)
            continue
    
    imgs = np.array(imgs)

    print('model structure', model)
    print('problem sub', problem_sub)
    print('data shape', imgs.shape)

    # predict
    for i in range(len(imgs)):
        # reconstruct the image
        _, reconstructed_img = model(torch.Tensor([imgs[i]]))
        reconstructed_img = reconstructed_img.detach().numpy().astype('float64')[0] # 1*91*109*91
        
        # calculate the residual
        residual_img = np.abs(imgs[i] - reconstructed_img) #1*91*109*91

        # thresholding
        # TODO

        # save image
        test_img = nib.Nifti1Image(reconstructed_img[0], affines[i], headers[i])
        test_img.to_filename('output/test_pred/reconstruction/{}_reconstruct.nii'.format(subjects[i]))

        residual_img = nib.Nifti1Image(residual_img[0], affines[i], headers[i])
        residual_img.to_filename('output/test_pred/residual/{}_residual.nii'.format(subjects[i]))