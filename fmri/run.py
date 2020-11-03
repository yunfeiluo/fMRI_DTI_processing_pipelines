'''
Project: fMRI processing pipelines
Author: Yunfei Luo
Start-Date: Jul 9th 2020
Last-modified: Jul 30th 2020
'''
import os
from multiprocessing import Pool
import nibabel as nib

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe

from pipelines.preprocess.pipeline import pipeline as preprocess_pipeline
from pipelines.first_level_analysis.pipeline import pipeline as level1_analysis_pipeline
from pipelines.integrate_pipeline import pipeline as int_pipeline

# specify subject names, and pipeline name
subjects = ['ZF36621', 'ZF49582', 'ZF53093', 'Burke', 'Moss-Einstein', 'MUSC', 'NationalRehabHospital', 'UK_Kentucky', 'Texas', 'UAB_Birmingham', 'USC_July9th_SecondAttempt', 'ZF18762', 'ZF44516', 'ZF59543', 'Emory']

problem_subjects = ['MUSC', 'NationalRehabHospital']
problem_corr_subjects = ['Emory', 'ZF21147', 'ZF24852', 'ZF39497', 'ZF43014']

aug_31_add = ['ZF10843', 'ZF19833', 'ZF36197']

oct_22_add = ['ZF17501', 'ZF22922', 'ZF24781', 'ZF26885', 'ZF44071', 'ZF48602', 'ZF52549']

# specify directories, and data template
experiment_dir = "tasks/ZFs_Oct_22/" # specify the root folder for tasks
output_dir = experiment_dir + "output/"

# define function for proprocess pipeline/workflow
def preprocess(subject):
    # define input node
    func_in = pe.Node(nio.DataGrabber(), name='functional_input_node')
    func_in.inputs.base_directory = '/mnt/Documents/data/ZFs_Oct_19_2020_nii'
    func_in.inputs.template = '{}/rsfmri/{}_rsfmri.nii'.format(subject, subject)
    func_in.inputs.sort_filelist = True
    func_img = nib.load(func_in.inputs.base_directory + '/{}/rsfmri/{}_rsfmri.nii'.format(subject, subject))
    TR = func_img.header['pixdim'][4]
    num_slices = func_img.shape[-2]
    num_vol = func_img.shape[-1]
    # TR, num_slices = 3.0, 120

    struct_in = pe.Node(nio.DataGrabber(), name='structual_input_node')
    struct_in.inputs.base_directory = '/mnt/Documents/data/ZFs_Oct_19_2020_nii/'
    struct_in.inputs.template = '{}/anat/{}_struct.nii'.format(subject, subject)
    # struct_in.inputs.base_directory = '/mnt/Program/python/dev/tasks/stroke_test/data/'
    # struct_in.inputs.template = 'sub{}.nii'.format(subject)
    struct_in.inputs.sort_filelist = True

    # define output node for preprocess
    preprocess_datasink = pe.Node(nio.DataSink(), name="preprocess_output_node")
    preprocess_datasink.inputs.base_directory = os.path.abspath(output_dir + 'preprocess_out/' + subject)
    
    # construct pipelines
    preprocess = preprocess_pipeline(
        subject=subject,
        func_source=func_in, 
        struct_source=struct_in, 
        datasink=preprocess_datasink,
        TR=TR,
        num_slices=num_slices)

    # run pipeline
    preprocess.workflow.write_graph()
    preprocess.workflow.run()
    return '############## complete subject {} #################'.format(subject)

if __name__ == '__main__':
    # arr = aug_31_adda
    # arr = [str(i+1) for i in range(26)]
    # arr = ['ZF59512']
    # arr = [str(i+1) for i in range(1)]
    arr = oct_22_add[1:2]
    p = Pool(processes=min(7, len(arr)))
    print('start processing {}'.format(arr))

    # run in parallel
    print(p.map(preprocess, arr))