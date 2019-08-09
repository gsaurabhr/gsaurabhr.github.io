# Notebooks by topic

## Differentiation in CNNs

### August

1. [8/9/2019 **The differentiation puzzle**](pages/VGG16_understand_differentiation/VGG16_understand_differentiation.md)
   In this notebook, I show the internal mean filter activations in the different layers of VGG16, when exposed to a set of noise, cat and random images. The patterns in the filter activations show variations that we expect from our basic understanding of a CNN - lower layers encode local features, higher layers encode semantic information. However, this is not reflected in the differentiation that we compute by randomly sampling units (200 units, or even a large number, 10000, of units). I use an **updated normalization** of activation and distance calculations, so that the **Euclidean distance metric is now equivalent to correlation distance metric**.

1. [8/6/2019 **Preliminary observations for differentiation in a CNN**](pages/VGG16_Differentiation_original/VGG16_Differentiation_original.md)
   Interestingly, differentiation as we calculate here does not show the trends that we would expect.