'''
Project: fMRI processing pipelines
Function: integrate the pre-preocessing and analysis pipeline
Author: Yunfei Luo
Start-Date: Jul 9th 2020
Last-modified: Jul 28th 2020
'''
import os

# warped software packages
import nipype.interfaces.io as nio

import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

class pipeline:
    def __init__(self, preprocess_pipeline, level1_analysis_pipeline):
        self.workflow = pe.Workflow(name='process_pipeline')
        # connect pipelines
        self.workflow.connect([
            (preprocess_pipeline, level1_analysis_pipeline, [('smooth.smoothed_files', 'model_specification.functional_runs')]),
            (preprocess_pipeline, level1_analysis_pipeline, [('realign_motion_correction.realignment_parameters', 'model_specification.realignment_parameters')])
        ])

    # Execute the node(s)
    def forward(self):
        self.workflow.run()
    
