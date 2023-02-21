import numpy as np
import nibabel as nib

from ResNet18AE import *
from ResNet34AE import *

if __name__ == '__main__':
    img = nib.load('MNI152_T1_1mm_brain.nii')
    img = img.get_fdata()

    model = Res34Autoencoder(begin_channel=64)
    x = torch.Tensor([img.reshape(1, 182, 218, 182) for i in range(1)])
    encode_out, y = model(x)
    # y, ind = model(x)
    print('x shape', x.shape)
    print('encode shape', encode_out.shape)
    print('y shape', y.shape)

    classifier = Res34Encoder(begin_channel=64, num_classes=2)
    y = classifier(x)
    print('class shape', y.shape)