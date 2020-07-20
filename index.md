# Notebooks by topic

## Visual Behavior

### January 2020

1. [01/10/2020 **Differentiation, PCIst during behavior task**](pages/behavior_1/behavior_1.md)  
   Mean firing rate, mfr_differentiation and PCIst metrics are computed for different regiona and layers, during the visual behavior task.

## Neuropixel recordings

### December 2019

1. [12/06/2019 **Differentiation by layer, differentiation in averaged responses**](pages/ephys_summary_1/ephys_summary_1.md)  
   Differentiation is computed by layer, and we see that it is significantly higher in superficial layers compared to deep (5/6) layers.  
   Independently, differentiation is computed on averaged responses to repeated movie stimuli, to see if that reduces the differentiation in the hippocampus, where activity is high, but not likely to be correlated with visual stimulus.

### November 2019

1. [11/15/2019 **Normalizing differentiation by number and activity**](pages/ephys_5_1/ephys_5_1.md)  
   Initially we were normalizing differentiation wrt spontaneous activity, in order to get rid of the effects of 

1. [11/01/2019 **Ephys - CNN comparisons (Showcase poster data)**](pages/showcase2019/showcase2019.md)  

### September 2019

1. [9/27/2019 **Differentiation (SD-ND correlation, smaller states)**](pages/ephys_4_2/ephys_4_2.md)  

1. [9/13/2019 **Differentiation analysis for Signal-Noise sessions**](pages/ephys_4_1/ephys_4_1.md)  
   Differentiation and SD vs ND correlation as a function of depth from two sessions with the Signal-Noise stimuli (these stimuli are monochrome NatGeo clips with mostly animal, landscape and some human videos.)

### August 2019

1. [8/28/2019 **Spectral differentiation of mean responses by session**](pages/ephys_3_2/ephys_3_2.md)  
   Also includes some results after changing analysis parameters to compare with Will's results.

1. [8/28/2019 **Spectral differentiation across ephys sessions - summary**](pages/ephys_3_1/ephys_3_1.md)  
   Mean differentiation for each stimulus as a function of hierarchy depth of region. Summarizes the notebooks below.

1. [8/28/2019 **Spectral differentiation across ephys sessions - 1**](pages/ephys_2_1/2_spectral_by_trial.md)  
   [8/28/2019 **Spectral differentiation across ephys sessions - 2**](pages/ephys_2_2/2_spectral_by_trial-session_2.md)  
   Spectral differentiation is computed across the entire session for two example sessions, for units from different brain areas and their supersets. Looking only at the differentiation during spontaneous activity, we find that differentiation varies quite a bit even in early visual areas (and also in higher areas), indicating that it is driven by something other than visual stimuli. We see that the correlation between running speed and differentiation explains some of this. Secondly, during stimulus presentations, we see that differentiation (normalized by that of spontaneous activity) varies a lot in (early) visual areas, but very little in higher areas. In fact, differentiation seems to be the same for stimuli and spontaneous activity in higher areas, which is very surprising.

1. [8/23/2019 **A first look at ephys data**](pages/ephys_1/1_Basics.md)  
   A preliminary look at neuropixel data shows some nice cases of units showing preferential activity to some kinds of stimuli. However, more importantly, the mean activity of a unit for repeated application of the same stimulus is very different from the trial to trial activity. This suggests that responses are extremely variable, and perhaps i should look deeper into the variability. In the meanwhile, this means that it is better to compute the differentiation an a trial by trial basis rather than on any kind of an averaged activity.

## Differentiation in CNNs

### September 2019

1. [9/13/2019 **SD vs ND for vision models (openscope stimuli)**](pages/CNNs_SD_ND_vgg_resnet_inception_hmax/CNNs_SD_ND_vgg_resnet_inception_hmax.md)
   Summary of results for HMAX, VGG16, ResNet50 and Inception V3 models, along with an example result from ephys data (see [here](pages/ephys_4_1/ephys_4_1.md) for details of corresponding npx analysis).

### August 2019

1. [8/13/2019 **VGG16 shown mouse stimuli**](pages/VGG16_mouse_stimuli/VGG16_mouse_stimuli.md)  
   Preliminary results (differentiation across layers for different movies, correlation between input image differentiation and activation differentiation in the layers, dependence of correlation on layer depth, stimulus set).

1. [8/12/2019 **Differentiation in test image sets**](pages/VGG16_understand_differentiation/VGG16_understand_differentiation.md)  
   In this notebook, I validate the differentiation analysis by applying it to CNN responses to three sets of images - _white noise_, _cats_ and _random_. The previous post showed some funny unexpected behavior, which I found was due to a bug, and has been fixed. I also use an **updated normalization** of activation and distance calculations, so that the **Euclidean distance metric is now equivalent to correlation distance metric**, which is more commonly used in such kind of RDM based analyses. Given that the observed differentiation makes sense, we can now apply it to the mouse stimuli! The notebook also shows filter activations and a few example filter representations from different layers.

1. [8/6/2019 **Preliminary observations for differentiation in a CNN**](pages/VGG16_Differentiation_original/VGG16_Differentiation_original.md)  
   Interestingly, differentiation as we calculate here does not show the trends that we would expect.

# Personal

1. [Biking recordings](pages/personal/alltracks.html)
