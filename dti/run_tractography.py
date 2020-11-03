'''
Project: fMRI processing pipelines
Author: Yunfei Luo
Start-Date: Sep 16th 2020
Last-modified: Sep 16th 2020
'''
import os
from multiprocessing import Pool
import nibabel as nib

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe

from src.pipelines.tractography import tractography

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
    # from preprocess pipelines
    dti_source = pe.Node(nio.DataGrabber(), name='dti_input_node')
    dti_source.inputs.base_directory = '/mnt/Program/python/dti/tasks/examples/output/preprocess_out'
    dti_source.inputs.template = '{}/bet_img/{}_brain.nii.gz'.format(subject, subject)
    dti_source.inputs.sort_filelist = True

    mask_source = pe.Node(nio.DataGrabber(), name='bet_mask_input_node')
    mask_source.inputs.base_directory = '/mnt/Program/python/dti/tasks/examples/output/preprocess_out'
    mask_source.inputs.template = '{}/bet_brain_mask/{}_brain_mask.nii.gz'.format(subject, subject)
    mask_source.inputs.sort_filelist = True

    # original files
    bval_source = pe.Node(nio.DataGrabber(), name='bval_input_node')
    bval_source.inputs.base_directory = '/mnt/Program/python/dti/imgs'
    bval_source.inputs.template = '{}/bvals'.format(subject)
    bval_source.inputs.sort_filelist = True

    bvec_source = pe.Node(nio.DataGrabber(), name='bvec_input_node')
    bvec_source.inputs.base_directory = '/mnt/Program/python/dti/imgs'
    bvec_source.inputs.template = '{}/bvecs'.format(subject)
    bvec_source.inputs.sort_filelist = True

    # external files
    seed_source = pe.Node(nio.DataGrabber(), name='seed_input_node')
    seed_source.inputs.base_directory = '/mnt/Program/python/dti/imgs'
    seed_source.inputs.template = 'filename_here'
    seed_source.inputs.sort_filelist = True

    # define output node for preprocess
    preprocess_datasink = pe.Node(nio.DataSink(), name="tractography_output_node")
    preprocess_datasink.inputs.base_directory = os.path.abspath(output_dir + 'tractography_out/' + subject)
    
    # construct pipelines
    pipeline = tractography(
        subject=subject,
        dti_source=dti_source, 
        mask_source=mask_source,
        bval_source=bval_source,
        bvec_source=bvec_source,
        seed_source=seed_source,
        datasink=preprocess_datasink,)

    # run pipeline
    pipeline.workflow.write_graph()

    try:
        pipeline.forward()
        return '############## complete subject {} #################'.format(subject)
    except:
        return '############## Failed subject {} #################'.format(subject)

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