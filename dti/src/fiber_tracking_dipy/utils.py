# import packages
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from multiprocessing import Pool
from scipy.ndimage.morphology import binary_dilation
import pickle

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere, small_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data

from dipy.direction import peaks, DeterministicMaximumDirectionGetter, ProbabilisticDirectionGetter
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel

from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion, BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor, colormap, has_fury

from dipy.io.utils import (create_nifti_header, get_reference_info,
                           is_header_compatible)
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.utils import density_map

print('import complete.')

# declare functions
def get_file_names(subject):
    '''
    @param subject: string represents the subject name
    @return, the necessary filenames for fiber tracking
    '''
    # example data
    if subject == 'stanford_hardi':
        fname, bval_fname, bvec_fname = get_fnames(subject)
        label_fname = get_fnames('stanford_labels') # need modification
        return fname, bval_fname, bvec_fname, label_fname

    # subejcts
    path_prefix = "/home/yunfeiluo/Documents/data/TRANSPORT2_WebDCU/{}/dti/{}_DTI".format(subject, subject)
    fname = path_prefix + ".nii"
    bval_fname = path_prefix + ".bval"
    bvec_fname = path_prefix + ".bvec"
    label_fname = None # need preprocessing
    
    return fname, bval_fname, bvec_fname, label_fname
    
def fiber_tracking(subject):
    # declare the type of algorithm, \in [deterministic, probabilitic]
    algo = 'deterministic'
#     algo = 'probabilitic'
    '''
    @param subject: string represents the subject name
    @param algo: the name for the algorithms, \in ['deterministic', 'probabilitic']
    @return streamlines: for saving the final results and visualization
    '''
    
    print('processing for', subject)
    fname, bval_fname, bvec_fname, label_fname = get_file_names(subject)

    data, sub_affine, img = load_nifti(fname, return_img=True)
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs)
    labels = load_nifti_data(label_fname)

    print('data loading complete.\n')
    ##################################################################
    
    # set mask(s) and seed(s)
    # global_mask = binary_dilation((data[:, :, :, 0] != 0))
    global_mask = binary_dilation((labels == 1) | (labels == 2))
#     global_mask = binary_dilation((labels == 2) | (labels == 32) | (labels == 76))
    affine = np.eye(4)
    seeds = utils.seeds_from_mask(global_mask, affine, density=1)
    print('mask(s) and seed(s) set complete.\n')
    ##################################################################
    
    print('getting directions from diffusion dataset...') 
    
    # define tracking mask with Constant Solid Angle (CSA)
    csamodel = CsaOdfModel(gtab, 6)
    stopping_criterion = BinaryStoppingCriterion(global_mask)
    
    # define direction criterion
    direction_criterion = None
    print('Compute directions...')
    if algo == "deterministic":
        # EuDX
        direction_criterion = peaks.peaks_from_model(model=csamodel,
                                  data=data,
                                  sphere=peaks.default_sphere,
                                  relative_peak_threshold=.8,
                                  min_separation_angle=45,
                                  mask=global_mask)

#         # Deterministic Algorithm (select direction with max probability)
#         direction_criterion = DeterministicMaximumDirectionGetter.from_shcoeff(
#             csd_fit.shm_coeff, 
#             max_angle=30., 
#             sphere=default_sphere)        
    else:
        response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
        
        # fit the reconstruction model with Constrained Spherical Deconvolusion (CSD)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
        csd_fit = csd_model.fit(data, mask=global_mask)
        
#         gfa = csamodel.fit(data, mask=global_mask).gfa
        #     stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
    
        # Probabilitic Algorithm
        direction_criterion = ProbabilisticDirectionGetter.from_shcoeff(
            csd_fit.shm_coeff,
            max_angle=30.,
            sphere=default_sphere)

    print('direction computation complete.\n')
    ##################################################################
    
    print('start tracking process...')
    # start tracking
    streamline_generator = LocalTracking(direction_criterion, stopping_criterion, seeds,
                                         affine=affine, step_size=0.5)
    
    # Generate streamlines object
    streamlines = Streamlines(streamline_generator)

    sft = StatefulTractogram(streamlines, img, Space.RASMM)
    
    print('traking complete.\n')
    ##################################################################

    return {"subject": subject, 
            "streamlines": streamlines, 
            "sft": sft, 
            "affine": sub_affine, 
            "data": data, 
            "img": img, 
            "labels": labels}

print('functions compling complete.')

def save_results(table, area_pairs, medium, subject_name):
    with open('saved/{}_track_result'.format(subject_name), 'wb') as f:
        pickle.dump(table, f)
        
    streamlines = table['streamlines']
    sft = table['sft']
    sub_affine = table['affine']
    data = table['data']
    labels = table["labels"]
    
    affine = np.eye(4)
    
    # extract streamlines only pass through ROIs
    cc_slice = (labels == medium[0]) # ROIs
    if len(medium) > 1:
        for m in medium:
            cc_slice = cc_slice | (labels == m)
    
    cc_streamlines = utils.target(streamlines, affine, cc_slice)
    cc_streamlines = Streamlines(cc_streamlines)

    other_streamlines = utils.target(streamlines, affine, cc_slice,
                                     include=False)
    other_streamlines = Streamlines(other_streamlines)
    assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)
    
    M, grouping = utils.connectivity_matrix(streamlines, affine, labels,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
    M[:3, :] = 0
    M[:, :3] = 0
    
    for pair in area_pairs:
        track = grouping[pair[0], pair[1]]
        shape = labels.shape
        dm = utils.density_map(track, affine, shape)

        import nibabel as nib

        # Save density map
        dm_img = nib.Nifti1Image(dm.astype("int16"), sub_affine)
        dm_img.to_filename("saved/{}_{}.nii.gz".format(subject_name, pair))

#     save_trk(sft, "tractogram_probabilistic_dg.trk")

#     # visualzie
#     # Enables/disables interactive visualization
#     interactive = False

#     if has_fury:
#         # Prepare the display objects.
#         color = colormap.line_colors(streamlines)

#         streamlines_actor = actor.line(streamlines, color)

#         # Create the 3D display.
#         r = window.Scene()
#         r.add(streamlines_actor)

#         # Save still images for this static example. Or for interactivity use
#         window.record(r, out_path='fiber_tracking_Caudate_l_r_result.png', size=(800, 800))

#         if interactive:
#             window.show(r)