'''
Project: fMRI pre-processing pipelines
Author: Yunfei Luo
Start-Date: Jul 9th 2020
Last-modified: Jul 13th 2020
'''
import os
from multiprocessing import Pool

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe

from pipelines.FuNP.pipeline import pipeline as FuNP_pipeline

workflow_plotted = False

# specify subject names, and pipeline name
subjects = ['sub001', 'sub002', 'sub003']
pipeline_name = 'FuNP'
chosen_pipeline = FuNP_pipeline

# specify directories, and data template
experiment_dir = "tasks/flanker_task/" # specify the root folder for tasks
output_dir = experiment_dir + "output/" + pipeline_name + "/"

# define function for proprocess pipeline/workflow
def preprocess(subject):
    # define input node
    func_in = pe.Node(nio.DataGrabber(), name='functional_input_node')
    func_in.inputs.base_directory = '/mnt/' + experiment_dir + 'data'
    func_in.inputs.template = subject + '/run*.nii'
    func_in.inputs.sort_filelist = True

    struct_in = pe.Node(nio.DataGrabber(), name='structual_input_node')
    struct_in.inputs.base_directory = '/mnt/' + experiment_dir + 'data'
    struct_in.inputs.template = subject + '/struct.nii'
    struct_in.inputs.sort_filelist = True

    # define output node
    datasink = pe.Node(nio.DataSink(), name="output_node")
    datasink.inputs.base_directory = os.path.abspath(output_dir + subject)

    # construct pipeline
    pipeline = chosen_pipeline(
        experiment_dir=experiment_dir, 
        output_dir = output_dir, 
        func_source = func_in, 
        struct_source = struct_in, 
        datasink = datasink)

    # # plot the workflow
    if not workflow_plotted:
        pipeline.workflow.write_graph()
        workflow_plotted = True

    # run pipeline
    pipeline.forward(subject)
    return '############## complete subject {} #################'.format(subject)

if __name__ == '__main__':
    p = Pool(processes=3)
    print(p.map(preprocess, subjects))