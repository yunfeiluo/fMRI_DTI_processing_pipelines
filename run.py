'''
Project: fMRI pre-processing pipelines
Author: Yunfei Luo
Start-Date: Jul 9th 2020
Last-modified: Jul 13th 2020
'''
import os

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe

from pipelines.spm_normal.pipeline import pipeline as pipeline0
from pipelines.FuNP.pipeline import pipeline as pipeline1

if __name__ == '__main__':
    # specify subject names, and pipeline name
    subjects = ['sub001']
    pipeline_name = 'FuNP'
    pl = pipeline1
    #pipeline_name = 'spm_normal'
    #pl = pipeline0

    # specify directories, and data template
    experiment_dir = "tasks/flanker_task/" # specify the root folder for tasks
    output_dir = experiment_dir + "output/" + pipeline_name + "/"
    working_dir = experiment_dir + "tmp/"+ pipeline_name + "/"

    # run pipelines on subjects
    for subject in subjects:
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
        pipeline = pl(
            experiment_dir=experiment_dir,
            output_dir = output_dir,
            working_dir = working_dir,
            func_source = func_in,
            struct_source = struct_in,
            datasink = datasink
        )

        # run pipeline
        pipeline.workflow.write_graph()
        pipeline.forward(subject)