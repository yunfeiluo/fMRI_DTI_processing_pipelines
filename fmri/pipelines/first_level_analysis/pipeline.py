'''
Project: fMRI processing pipelines
Function: First-Level_Analysis pipeline
Author: Yunfei Luo
Start-Date: Jul 9th 2020
Last-modified: Jul 28th 2020
'''
import os

# warped software packages
import nipype.interfaces.io as nio
from nipype.interfaces import fsl, spm
import nipype.algorithms.modelgen as model

import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
from nipype.interfaces.base import Bunch

import nibabel as nib

class pipeline:
    def __init__(self, datasink, TR, num_vol):
        # specify input and output nodes
        self.datasink = datasink
        self.TR = TR
        self.num_vol = num_vol

        # specify nodes
        # SpecifyModel - Generates SPM-specific Model
        self.modelspec = pe.Node(interface=model.SpecifySPMModel(), name='model_specification')
        self.modelspec.inputs.input_units = 'secs'
        self.modelspec.inputs.output_units = 'secs'
        self.modelspec.inputs.time_repetition = self.TR
        self.modelspec.inputs.high_pass_filter_cutoff = 128
        subjectinfo = [Bunch(conditions=['None'], onsets=[list(range(self.num_vol))], durations=[[0.5]])]
        self.modelspec.inputs.subject_info = subjectinfo

        # Level1Design - Generates an SPM design matrix
        self.level1design = pe.Node(interface=spm.Level1Design(), name='first_level_design')
        self.level1design.inputs.bases = {'hrf': {'derivs': [1, 1]}}
        self.level1design.inputs.interscan_interval = self.TR
        self.level1design.inputs.timing_units = 'secs'

        # EstimateModel - estimate the parameters of the model
        # method can be 'Classical', 'Bayesian' or 'Bayesian2'
        self.level1estimate = pe.Node(interface=spm.EstimateModel(), name="first_level_estimate")
        self.level1estimate.inputs.estimation_method = {'Classical': 1}

        self.threshold = pe.Node(interface=spm.Threshold(), name="threshold")
        self.threshold.inputs.contrast_index = 1

        # EstimateContrast - estimates contrasts
        self.contrast_estimate = pe.Node(interface=spm.EstimateContrast(), name="contrast_estimate")
        cont1 = ('active > rest', 'T', ['None'], [1])
        contrasts = [cont1]
        self.contrast_estimate.inputs.contrasts = contrasts

        # specify workflow instance
        self.workflow = pe.Workflow(name='first_level_analysis_workflow')

        # connect nodes
        self.workflow.connect([
            (self.modelspec, self.level1design, [('session_info', 'session_info')]),
            (self.level1design, self.level1estimate, [('spm_mat_file', 'spm_mat_file')]),
            (self.level1estimate, self.contrast_estimate, [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'), ('residual_image', 'residual_image')]),
            # (self.contrast_estimate, self.threshold, [('spm_mat_file', 'spm_mat_file'), ('spmT_images', 'stat_image')]),
            (self.contrast_estimate, self.datasink, [('con_images', 'contrast_img'), ('spmT_images', 'contrast_T')])
        ])

    # Execute the node(s)
    def forward(self):
        self.workflow.run()
    
