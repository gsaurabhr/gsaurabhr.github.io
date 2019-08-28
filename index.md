# Notebooks by topic

## Neuropixel recordings

### August

1. [8/28/2019 **Spectral differentiation across ephys sessions - 1**](pages/ephys_2_1/2_spectral_by_trial.md)  
   [8/28/2019 **Spectral differentiation across ephys sessions - 2**](pages/ephys_2_2/2_spectral_by_trial-session_2.md)
   Spectral differentiation is computed across the entire session for two example sessions, for units from different brain areas and their supersets. Looking only at the differentiation during spontaneous activity, we find that differentiation varies quite a bit even in early visual areas (and also in higher areas), indicating that it is driven by something other than visual stimuli. We see that the correlation between running speed and differentiation explains some of this. Secondly, during stimulus presentations, we see that differentiation (normalized by that of spontaneous activity) varies a lot in (early) visual areas, but very little in higher areas. In fact, differentiation seems to be the same for stimuli and spontaneous activity in higher areas, which is very surprising.

1. [8/23/2019 **A first look at ephys data**](pages/ephys_1/1_Basics.md)
   A preliminary look at neuropixel data shows some nice cases of units showing preferential activity to some kinds of stimuli. However, more importantly, the mean activity of a unit for repeated application of the same stimulus is very different from the trial to trial activity. This suggests that responses are extremely variable, and perhaps i should look deeper into the variability. In the meanwhile, this means that it is better to compute the differentiation an a trial by trial basis rather than on any kind of an averaged activity.

## Differentiation in CNNs

### August

1. [8/13/2019 **VGG16 shown mouse stimuli**](pages/VGG16_mouse_stimuli/VGG16_mouse_stimuli.md)
   Preliminary results (differentiation across layers for different movies, correlation between input image differentiation and activation differentiation in the layers, dependence of correlation on layer depth, stimulus set).

1. [8/12/2019 **Differentiation in test image sets**](pages/VGG16_understand_differentiation/VGG16_understand_differentiation.md)
   In this notebook, I validate the differentiation analysis by applying it to CNN responses to three sets of images - _white noise_, _cats_ and _random_. The previous post showed some funny unexpected behavior, which I found was due to a bug, and has been fixed. I also use an **updated normalization** of activation and distance calculations, so that the **Euclidean distance metric is now equivalent to correlation distance metric**, which is more commonly used in such kind of RDM based analyses. Given that the observed differentiation makes sense, we can now apply it to the mouse stimuli! The notebook also shows filter activations and a few example filter representations from different layers.

1. [8/6/2019 **Preliminary observations for differentiation in a CNN**](pages/VGG16_Differentiation_original/VGG16_Differentiation_original.md)
   Interestingly, differentiation as we calculate here does not show the trends that we would expect.
