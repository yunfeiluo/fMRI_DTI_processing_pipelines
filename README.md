# fMRI_DTI_processing_pipelines 
fMRI data processing pipelines  

# Components 
## fMRI pre-processing pipeline 
- skull strip  
- motion correction  
- registration  
- segmentation  
- normalization  
## DTI processing pipeline 
- normalization
- DIPY fiber tracking
- FDT tracktography pipeline

# dependencies 
- nipype, https://github.com/nipy/nipype 
- spm12, https://www.fil.ion.ucl.ac.uk/spm/software/spm12/ 
- fsl, https://fsl.fmrib.ox.ac.uk/fsl/fslwiki 
- afni 
- DIPY  

# reference 
- spm12 manual 
- FuNP, https://www.frontiersin.org/articles/10.3389/fninf.2019.00005/full  
- Density map, https://dipy.org/documentation/1.0.0./examples_built/streamline_tools/

---

# GAN_for_Stroke_Detection  

# Decription 
- Generative Adversarial Network for Stroke Detection  
- Unsupervised Learning Approach  

# Models' Structure  
- Convolutional Autoencoder as Generator, with ResNet structure  
- ResNet classifier as Discriminator  

# Dataset(s)  
- Train on Healthy data  
- Flanker task, with Referece  
Kelly, A.M., Uddin, L.Q., Biswal, B.B., Castellanos, F.X., Milham, M.P. (2008). Competition between functional brain networks mediates behavioral variability. Neuroimage, 39(1):527-37  
