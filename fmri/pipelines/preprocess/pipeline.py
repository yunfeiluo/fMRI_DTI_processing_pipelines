'''
Project: fMRI processing pipelines
Function: preprocessing pipeline
Author: Yunfei Luo
Start-Date: Jul 9th 2020
Last-modified: Jul 28th 2020
'''
import os

# warped software packages
import nipype.interfaces.io as nio
from nipype.interfaces import fsl, spm

import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

class pipeline:
    def __init__(self, subject, func_source, struct_source, datasink, TR, num_slices, dim=2):
        self.subject = subject
        # specify input and output nodes
        self.func_source = func_source
        self.struct_source = struct_source
        self.datasink = datasink

        self.TR = TR
        self.num_slices = num_slices

        # specify nodes
        # structual process
        self.bet_struct = pe.Node(interface=fsl.BET(), name='non_brain_removal_BET_struct')
        self.bet_struct.inputs.output_type = "NIFTI"
        self.bet_struct.inputs.frac = 0.3

        self.coregister_struct = pe.Node(interface=spm.Coregister(), name="coregister_struct_to_mni")
        self.coregister_struct.inputs.target = '/mnt/Program/python/dev/images/MNI152_T1_2mm_brain.nii'
        if dim == 1:
            self.coregister_struct.inputs.target = '/mnt/Program/python/dev/images/MNI152_T1_1mm_brain.nii'
        self.coregister_struct.inputs.write_interp = 7
        self.coregister_struct.inputs.separation = [1.0, 1.0]
        self.coregister_struct.inputs.jobtype = 'estwrite'

        self.segment_struct = pe.Node(interface=spm.Segment(), name="segment_struct")
        # self.segment_struct.inputs.affine_regularization = 'mni'
        self.segment_struct.inputs.csf_output_type = [True,True,True]
        self.segment_struct.inputs.gm_output_type = [True,True,True]
        self.segment_struct.inputs.wm_output_type = [True,True,True]

        self.normalize_struct = pe.Node(interface=spm.Normalize(), name='normalize_struct')
        self.normalize_struct.inputs.jobtype = 'write'
        self.normalize_struct.inputs.write_bounding_box= [[-90, -126, -72], [90, 90, 108]] # 91, 109, 91
        if dim == 1:
            self.normalize_struct.inputs.write_bounding_box= [[-91, -126, -72], [90, 91, 109]] # 182, 218, 182
            self.normalize_struct.inputs.write_voxel_sizes = [1, 1, 1]
    
        self.normalize_gm = pe.Node(interface=spm.Normalize(), name='normalize_gm')
        self.normalize_gm.inputs.jobtype = 'write'
        self.normalize_gm.inputs.write_bounding_box= [[-90, -126, -72], [90, 90, 108]] # 91, 109, 91
        if dim == 1:
            self.normalize_gm.inputs.write_bounding_box= [[-91, -126, -72], [90, 91, 109]] # 182, 218, 182
            self.normalize_gm.inputs.write_voxel_sizes = [1, 1, 1]

        self.normalize_wm = pe.Node(interface=spm.Normalize(), name='normalize_wm')
        self.normalize_wm.inputs.jobtype = 'write'
        self.normalize_wm.inputs.write_bounding_box= [[-90, -126, -72], [90, 90, 108]] # 91, 109, 91
        if dim == 1:
            self.normalize_wm.inputs.write_bounding_box= [[-91, -126, -72], [90, 91, 109]] # 182, 218, 182
            self.normalize_wm.inputs.write_voxel_sizes = [1, 1, 1]

        self.normalize_csf = pe.Node(interface=spm.Normalize(), name='normalize_csf')
        self.normalize_csf.inputs.jobtype = 'write'
        self.normalize_csf.inputs.write_bounding_box= [[-90, -126, -72], [90, 90, 108]] # 91, 109, 91
        if dim == 1:
            self.normalize_csf.inputs.write_bounding_box= [[-91, -126, -72], [90, 91, 109]] # 182, 218, 182
            self.normalize_csf.inputs.write_voxel_sizes = [1, 1, 1]

        ###################################################################################################

        # functional process
        self.fslsplit = pe.Node(interface=fsl.Split(), name='fslsplit')
        self.fslsplit.inputs.dimension = 't'
        self.fslsplit.inputs.output_type = "NIFTI"

        self.fslmerge = pe.Node(interface=fsl.Merge(), name='fslmerge')
        self.fslmerge.inputs.dimension = 't'
        self.fslmerge.inputs.output_type = "NIFTI"
        
        # helper function(s)
        def bet_each(in_files, subject_name):
            '''
            @param in_files: list of image files
            @return out_files: list of image files after applied fsl.BET on it
            '''
            from nipype.interfaces import fsl
            import nipype.pipeline.engine as pe

            out_files = list()
            step_no = 0
            for file_ in in_files:
                bet = pe.Node(interface=fsl.BET(), name='BET_for_step_{}_{}'.format(step_no, subject_name))
                bet.inputs.in_file = file_
                bet.inputs.out_file = file_[:len(file_) - 4] + '_bet.nii'
                bet.inputs.output_type = "NIFTI"
                bet.inputs.frac = 0.5

                bet.run()
                out_files.append(bet.inputs.out_file)

                step_no += 1
            return out_files
        # bet_func return a list of NIFITI files
        self.bet_func = pe.Node(interface=Function(input_names=['in_files', 'subject_name'], output_names=['out_files'], function=bet_each), name='non_brain_removal_BET_func')
        self.bet_func.inputs.subject_name = self.subject

        self.realign = pe.Node(interface=spm.Realign(), name='realign_motion_correction')
        self.realign.inputs.register_to_mean = True

        self.coregister_func = pe.Node(interface=spm.Coregister(), name="coregister_func_to_mni")
        self.coregister_func.inputs.target = '/mnt/Program/python/dev/images/MNI152_T1_2mm_brain.nii'
        self.coregister_func.inputs.write_interp = 7
        self.coregister_func.inputs.separation = [1.0, 1.0]
        self.coregister_func.inputs.jobtype = 'estwrite'

        self.segment = pe.Node(interface=spm.Segment(), name="segment")

        self.normalize_func = pe.Node(interface=spm.Normalize(), name="normalize_func")
        self.normalize_func.inputs.jobtype = 'write'
        self.normalize_func.inputs.write_bounding_box= [[-90, -126, -72], [90, 90, 108]] # 91, 109, 91

        self.smooth = pe.Node(interface=spm.Smooth(), name="smooth")
        self.smooth.inputs.fwhm = [8, 8, 8]

        # backup node(s)
        self.slice_timing = pe.Node(interface=spm.SliceTiming(), name='time_slice_correction')
        self.slice_timing.inputs.time_repetition = self.TR
        self.slice_timing.inputs.num_slices = self.num_slices
        self.slice_timing.inputs.time_acquisition = self.TR - (self.TR / self.num_slices)
        self.slice_timing.inputs.slice_order = list(range(self.num_slices, 0, -1))
        self.slice_timing.inputs.ref_slice = 1

        self.direct_normalize = pe.Node(interface=spm.Normalize12(), name='direct_normalize')
        self.direct_normalize.inputs.image_to_align = 'images/MNI152_T1_2mm_brain.nii'
        self.direct_normalize.inputs.affine_regularization_type = 'size'

        # specify workflow instance
        self.workflow = pe.Workflow(name='preprocess_workflow')

        # connect nodes
        self.workflow.connect([
            (self.struct_source, self.bet_struct, [('outfiles', 'in_file')]),
            (self.bet_struct, self.coregister_struct, [('out_file', 'source')]),
            (self.coregister_struct, self.segment_struct, [('coregistered_source', 'data')]),
            (self.segment_struct, self.normalize_struct, [('transformation_mat', 'parameter_file')]),
            (self.coregister_struct, self.normalize_struct, [('coregistered_source', 'apply_to_files')]), 
            (self.segment_struct, self.normalize_gm, [('transformation_mat', 'parameter_file')]),
            (self.segment_struct, self.normalize_gm, [('native_gm_image', 'apply_to_files')]),
            (self.segment_struct, self.normalize_wm, [('transformation_mat', 'parameter_file')]),
            (self.segment_struct, self.normalize_wm, [('native_wm_image', 'apply_to_files')]),
            (self.segment_struct, self.normalize_csf, [('transformation_mat', 'parameter_file')]),
            (self.segment_struct, self.normalize_csf, [('native_csf_image', 'apply_to_files')]),
            (self.func_source, self.fslsplit, [('outfiles', 'in_file')]),
            (self.fslsplit, self.bet_func, [('out_files', 'in_files')]),
            (self.bet_func, self.fslmerge, [('out_files', 'in_files')]),
            (self.fslmerge, self.realign, [('merged_file', 'in_files')]),
            (self.realign, self.datasink, [('realignment_parameters', 'realignment_parameters')]),
            (self.realign, self.coregister_func, [('mean_image', 'source')]),
            (self.realign, self.coregister_func, [('realigned_files', 'apply_to_files')]),
            (self.coregister_func, self.segment, [('coregistered_source', 'data')]),
            (self.segment, self.normalize_func, [('transformation_mat', 'parameter_file')]),
            (self.coregister_func, self.normalize_func, [('coregistered_files', 'apply_to_files')]),
            (self.normalize_func, self.smooth, [('normalized_files', 'in_files')]), # end
            (self.normalize_func, self.datasink, [('normalized_files', 'before_smooth')]),
            (self.smooth, self.datasink, [('smoothed_files', 'final_out')]),
            (self.normalize_struct, self.datasink, [('normalized_files', 'standardized_struct_file')]),
            # (self.segment_struct, self.datasink, [('native_csf_image', 'csf'), ('native_gm_image', 'grey_matter'), ('native_wm_image', 'white_matter')])
            (self.normalize_gm, self.datasink, [('normalized_files', 'grey_matter')]),
            (self.normalize_wm, self.datasink, [('normalized_files', 'white_matter')]),
            (self.normalize_csf, self.datasink, [('normalized_files', 'csf')]),
        ])

        # backup workflow(s)
        self.workflow_only_fmri = pe.Workflow(name='preprocess_workflow')

        # connect nodes
        self.workflow_only_fmri.connect([
            (self.func_source, self.fslsplit, [('outfiles', 'in_file')]),
            (self.fslsplit, self.bet_func, [('out_files', 'in_files')]),
            (self.bet_func, self.fslmerge, [('out_files', 'in_files')]),
            (self.fslmerge, self.realign, [('merged_file', 'in_files')]),
            (self.realign, self.direct_normalize, [('realigned_files', 'apply_to_files')]),
            (self.direct_normalize, self.smooth, [('normalized_files', 'in_files')]), # end
            (self.direct_normalize, self.datasink, [('normalized_files', 'before_smooth')]),
            (self.smooth, self.datasink, [('smoothed_files', 'final_out')])
        ])


    # Execute the node(s)
    def forward(self):
        self.workflow.run()
    
