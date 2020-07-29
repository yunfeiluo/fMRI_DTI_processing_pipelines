'''
Project: fMRI processing pipelines
Author: Yunfei Luo
Start-Date: Jul 9th 2020
Last-modified: Jul 28th 2020
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
subjects = ['sub001', 'sub002', 'sub003']

# specify directories, and data template
experiment_dir = "tasks/flanker_task/" # specify the root folder for tasks
output_dir = experiment_dir + "output/"

# define function for proprocess pipeline/workflow
def launch_process(subject):
    # define input node (top nodes)
    func_in = pe.Node(nio.DataGrabber(), name='functional_input_node')
    func_in.inputs.base_directory = '/mnt/' + experiment_dir + 'data'
    func_in.inputs.template = subject + '/run*.nii'
    func_in.inputs.sort_filelist = True
    func_img = nib.load(func_in.inputs.base_directory + '/' + subject + '/run001.nii')
    TR = func_img.header['pixdim'][4]
    num_vol = func_img.shape[-1]

    struct_in = pe.Node(nio.DataGrabber(), name='structual_input_node')
    struct_in.inputs.base_directory = '/mnt/' + experiment_dir + 'data'
    struct_in.inputs.template = subject + '/struct.nii'
    struct_in.inputs.sort_filelist = True

    # define output node for preprocess
    preprocess_datasink = pe.Node(nio.DataSink(), name="preprocess_output_node")
    preprocess_datasink.inputs.base_directory = os.path.abspath(output_dir + 'preprocess_out/' + subject)

    # define output node for first level analysis
    level1_analysis_datasink = pe.Node(nio.DataSink(), name="first_level_analysis_output_node")
    level1_analysis_datasink.inputs.base_directory = os.path.abspath(output_dir + 'level1_analysis_out/' + subject)

    # construct pipelines
    preprocess = preprocess_pipeline(func_source=func_in, struct_source=struct_in, datasink=preprocess_datasink)
    level1_analysis = level1_analysis_pipeline(datasink=level1_analysis_datasink, TR=TR, num_vol=num_vol)

    # plot the workflow
    pipeline = int_pipeline(preprocess.workflow, level1_analysis.workflow)
    pipeline.workflow.write_graph()

    # run pipeline
    pipeline.forward()
    return '############## complete preprocess subject {} #################'.format(subject)

if __name__ == '__main__':
    p = Pool(processes=1)
    arr = subjects[:1]
    print(p.map(launch_process, arr))