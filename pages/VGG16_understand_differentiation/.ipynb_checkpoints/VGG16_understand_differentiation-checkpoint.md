
# Find sensible results with VGG16


```python
%%capture --no-stdout
%load_ext autoreload
%autoreload 2
import pickle
from tqdm import tqdm_notebook as tqdm
import PIL
from CNNDifferentiation import *
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input

cnn = CNNAnalysis(VGG16(weights='imagenet', include_top=True),
                  decode_predictions, preprocess_input, n_units=200)
```


```python
diffs = {200 : {}, 10000 : {}}
activations = {}
for p in ['../data/random_images/*.jpg', '../data/white_noise/*.jpg', '../data/tabby_cats/*.jpg']:
    cnn.load_images(p)
    for n_units in [200, 10000]:
        cnn.n_units = n_units
        cnn.compute_differentiation(resample=True)
        diffs[n_units][p.split('/')[-2]] = cnn.differentiation
    plt.figure(figsize=(20, 4))
    pred_acts = cnn.activations[-1] / cnn.activations[-1].mean()
    pred_acts = pred_acts / pred_acts.std()
    plt.imshow(cnn.activations[-1], vmin=0, vmax=1)
    plt.title('Activations in prediction layer for %s'%(p.split('/')[-2]))
    activations[p.split('/')[-2]] = cnn.activations
```
![png](output_2_13.png)



![png](output_2_14.png)



![png](output_2_15.png)


As can be seen, random images lead to peak activations in very different units. Cat images activate the same unit (and a few other units, weakly) in the prediction layer. White noise tend to systematically activate one of a few different units in prediction layer.

# Play with activations

## Mean filter activations


```python
index = {'white_noise' : 0, 'tabby_cats' : 1, 'random_images' : 2}
```


```python
for layer in range(cnn.n_layers):
    f, axes = plt.subplots(1, 3, figsize=(20, 3))
    for key in activations.keys():
        norm_acts = activations[key][layer] - activations[key][layer].mean()
        norm_acts = norm_acts / norm_acts.std()
        try:
            axes[index[key]].imshow(norm_acts.mean(axis=(1, 2)), vmin=-1, vmax=1)
        except:
            axes[index[key]].imshow(norm_acts, vmin=-1, vmax=1)
        axes[index[key]].set_title('Layer %s (%s)'%(cnn.layer_names[layer], key))
```



![png](output_7_1.png)



![png](output_7_2.png)



![png](output_7_3.png)



![png](output_7_4.png)



![png](output_7_5.png)



![png](output_7_6.png)



![png](output_7_7.png)



![png](output_7_8.png)



![png](output_7_9.png)



![png](output_7_10.png)



![png](output_7_11.png)



![png](output_7_12.png)



![png](output_7_13.png)



![png](output_7_14.png)



![png](output_7_15.png)



![png](output_7_16.png)



![png](output_7_17.png)



![png](output_7_18.png)



![png](output_7_19.png)



![png](output_7_20.png)



![png](output_7_21.png)


There are many observations to be made here, starting with the top panels and working downward:
1. In the images themselves (input layer) we do not see any vertical patterns (mean structure in R, G and B channels). This is a good sanity check, because our input images do not have a particularly large component of R or G or B. Interestingly, there is vertical structure in the noise images, which is unexpected and difficult to explain).
2. In the block1_conv* layers, we see that there are a few filters that show higher average activity than the rest, consistently for cats and random image sets. For instance, filters 31, 39 and 61 in block1_conv1. There are more such filters in conv2 and conv3 layers in block1.
3. For the same block1 layers, filters with high average activation for cats and random images are not necessarily active for noise. This is interesting, but perhaps makes sense because they are picking up low level structure and patterns such as edges, which are absent in the noise input.
4. Given that for real images, some filters are more active than others, we might get a stronger differentiation signal by sampling units from these filters since they actually contian information (in contrast to the large set of inactive filters that are presumably not carrying as much information).
5. We continue to see such vertical structure in the intermediate layers, with similar trends as discussed above for block1. However, to some extent in block 4, and more so in block 5, the vertical structure is arguably stronger for cats than random images. This might be the beginning of 'cat sensitive filters', which light up for cats, giving the vertical structure (ie high activation across all input images). In that sense, differentiation is now decreasing for cats, but is remaining highish for random images.
6. The strong vertical structure in white noise would imply that differentiation is very low for noise.
7. Somehow this signal is getting lost in our random sampling procedure, and it is not clear how best to recover it.

# Differentiation after normalizing activity
In the previous analysis, my normalization was not appropriate (I was normalizing activity so that the mean activity in each layer is 1). A better way to normalize is to set the mean activity to 0 and SD of activity to 1. With this normalization, the Euclidean distance metric becomes equivalent to correlation distance, which is more commonly used in ANN analyses.

We previously saw that differentiation behaved in the same way for noise, cats and random images. This was a little puzzling, but then I realized that it might be an artefact of sampling only 200 units. Activations in CNNs are sparse, and by sampling a very small fraction of units, I was probably seeing only the inactive units that do not matter anyway. That would explain the lack of difference in noise vs cats vs random image sets.

To see if this reasoning is indeed true, I repeated the analysis, but now, sampled 10,000 units from each layer. Moreover, in order to compare the distance metric with the 200 unit case, I normalized the distance by the number of dimensions, so that the value plotted is actually the mean distance per dimension. With this normalization, the Euclidean distance metric is now completely identical to the correlation distance metric.

We see that now, differentiation is nearly constant across all layers, including the input image.


```python
plt.figure(figsize=(16, 4))
plt.plot(range(cnn.n_layers), diffs[200]['random_images'], '--', label='200 random', c='g')
plt.plot(range(cnn.n_layers), diffs[10000]['random_images'], '-', label='10000 random', c='g')
plt.plot(range(cnn.n_layers), diffs[200]['white_noise'], '--', label='200 noise', c='r')
plt.plot(range(cnn.n_layers), diffs[10000]['white_noise'], '-', label='10000 noise', c='r')
plt.plot(range(cnn.n_layers), diffs[200]['tabby_cats'], '--', label='200 cats', c='b')
plt.plot(range(cnn.n_layers), diffs[10000]['tabby_cats'], '-', label='10000 cats', c='b')
plt.xlabel('Layer')
plt.ylabel('Differentiation')
plt.setp(plt.gca(), xticks=range(cnn.n_layers), xticklabels=cnn.layer_names)
plt.xticks(rotation=60)
plt.legend()
plt.grid()
```


![png](output_10_0.png)


The figure below shows the differentiation, normalized by differentiation in the noise set.

The only part that makes complete sense is what happens in the predictions layer. Here, when we sample only 200 units out of 1000, we probably miss the 'cat' unit, and most of the labels for random images, and mostly sample the remaining units which have a very low activity. Thus, differentiation for both the cat and random image sets is low.

However, when we sample all 1000 units form the prediction layer, we are now including the most important unit (cats), as well as all label units that occur in the random images. Thus, now, the differentiation drpos for cats, but jumps high for the random images, as can be seen.

It is interesting to note that differentiation does not show any other interesting signal even when we sample 10,000 units. As we saw from the activation heatmaps, we should be able to see interesting features in differentiation, and need to understand why we do not see them in our analysis, before applying it to the openscope stimuli.


```python
plt.figure(figsize=(16, 4))
plt.plot(range(cnn.n_layers), diffs[200]['random_images']/diffs[200]['white_noise'], '--', label='200 random', c='g')
plt.plot(range(cnn.n_layers), diffs[10000]['random_images']/diffs[10000]['white_noise'], '-', label='10000 random', c='g')
plt.plot(range(cnn.n_layers), diffs[200]['tabby_cats']/diffs[200]['white_noise'], '--', label='200 cats', c='b')
plt.plot(range(cnn.n_layers), diffs[10000]['tabby_cats']/diffs[10000]['white_noise'], '-', label='10000 cats', c='b')
plt.xlabel('Layer')
plt.ylabel('Differentiation')
plt.setp(plt.gca(), xticks=range(cnn.n_layers), xticklabels=cnn.layer_names)
plt.xticks(rotation=60)
plt.legend()
plt.ylim(0, 2)
plt.grid()
```


![png](output_12_0.png)



```python

```
