# Notebooks by topic

## Differentiation in CNNs

### August

1. [8/9/2019 **Differentiation in test image sets**](pages/VGG16_understand_differentiation/VGG16_understand_differentiation.md)
   In this notebook, I validate the differentiation analysis by applying it to CNN responses to three sets of images - _white noise_, _cats_ and _random_. The previous post showed some funny unexpected behavior, which I found was due to a bug, and has been fixed. I also use an **updated normalization** of activation and distance calculations, so that the **Euclidean distance metric is now equivalent to correlation distance metric**, which is more commonly used in such kind of RDM based analyses. Given that the observed differentiation makes sense, we can now apply it to the mouse stimuli! The notebook also shows filter activations and a few example filter representations from different layers.

1. [8/6/2019 **Preliminary observations for differentiation in a CNN**](pages/VGG16_Differentiation_original/VGG16_Differentiation_original.md)
   Interestingly, differentiation as we calculate here does not show the trends that we would expect.
