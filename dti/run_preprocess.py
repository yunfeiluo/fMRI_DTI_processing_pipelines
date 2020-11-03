'''
Project: fMRI processing pipelines
Author: Yunfei Luo
Start-Date: Sep 16th 2020
Last-modified: Sep 16th 2020
'''
import os
from multiprocessing import Pool
import nibabel as nib
import numpy as np

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe

from src.pipelines.preprocessing import ants_registr_FA

# specify subject names, and pipeline name
subjects = [
    'ZF36621', 
    'ZF44516', 
    'ZF49582', 
    'ZF53093', 
    'ZF59543', 
    'ZF18762', 
    'ZF21147', 
    'ZF24852', 
    'ZF39497', 
    'ZF43014',
    'ZF10843',
    'ZF18595',
    'ZF19833',
    'ZF30335',
    'ZF36197',
    'ZF36759',
    'ZF49381',
    'ZF55407',
    'ZF59512'
    ]

retry_subs = [
    'ZF10843', 
    'ZF21147',
    'ZF30335',
    'ZF39497',
    'ZF43014'
]

fuzzy_subs = [
    'ZF21147',
    'ZF39497',
    'ZF49381',
    'ZF55407'
]

# specify directories, and data template
experiment_dir = "tasks/examples/" # specify the root folder for tasks
output_dir = experiment_dir + "output/"

# define function for proprocess pipeline/workflow
def preprocess(subject):
    # define input node
    fa_source = pe.Node(nio.DataGrabber(), name='dti_fa_input_node')
    # fa_source.inputs.base_directory = '/mnt/Documents/data/TRANSPORT2_WebDCU'
    # fa_source.inputs.template = '{}/dti/{}_FA.nii'.format(subject, subject)
    fa_source.inputs.base_directory = '/mnt/Program/python/dti/imgs'
    fa_source.inputs.template = '{}/{}.nii'.format(subject, subject)
    fa_source.inputs.sort_filelist = True

    # define output node for preprocess
    preprocess_datasink = pe.Node(nio.DataSink(), name="preprocessing_output_node")
    preprocess_datasink.inputs.base_directory = os.path.abspath(output_dir + 'preprocess_out/' + subject)
    
    # construct pipelines
    pipeline = ants_registr_FA(
        subject=subject,
        fa_source=fa_source, 
        datasink=preprocess_datasink,)

    # run pipeline
    pipeline.workflow.write_graph()

    try:
        pipeline.forward()
        return '############## complete subject {} #################'.format(subject)
    except:
        return '############## Failed subject {} #################'.format(subject)

    # # output the brain mask
    # img = nib.load(output_dir + 'preprocess_out/' + subject + '/wASyN/transform_Warped.nii.gz')
    # affine = img.affine
    # header = img.header
    # img = img.get_fdata()
    # mask = (img > 0).astype('int')
    # mask = nib.Nifti1Image(mask, affine, header)
    # mask.to_filename(output_dir + 'preprocess_out/' + subject + '/brain_mask.nii')

if __name__ == '__main__':
    # arr = fuzzy_subs 
    arr = ['stanford_hardi']
    process_num = min(7, len(arr))
    p = Pool(processes=process_num)

    # run in parallel
    print('Num subjects', len(arr))
    print('Num processes', process_num)
    print('start processing {}'.format(arr))
    res_state = p.map(preprocess, arr)

    # print out result states
    for res in res_state:
        print(res)