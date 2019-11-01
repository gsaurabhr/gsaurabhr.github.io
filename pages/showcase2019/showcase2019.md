# Comparison of differentiation in CNNs and Neuropixel data
All data on this page is for differentiation calculated with a 0.1s state length, and 3s windows.

## Differentiation normalized to gabors vs spontaneous activity

### Does not make much difference for Neuropixel data
![png](sn_df_spontaneous.png)
![png](sn_df_gabor.png)

### But can still be tricky to interpret
![png](bo_df_spontaneous.png)
![png](bo_df_gabor.png)

### Especially true for our CNNs
![png](inception_df_mousenoise.png)
![png](inception_df_gabor.png)

Also note how gratings have a very high differentiation in early layers compared to the rest of stimuli. This is consistent with the understanding that early layers encode edges. This can get quite dramatic:

![png](vgg_df_gabor.png)

![png](resnet_df_gabor.png)

## SD vs ND in CNNs and Neuropixel data
![png](corr_resnet.png)
![png](corr_hmax.png)
![png](corr_vgg.png)

### Overall correlation is high in Npx data
![png](corr.png)

But this is because of the few extreme values in stimulus differentiation:
![png](sd_nd_scatter.png)

If we look only at the clustered regions:
![png](corr_selected.png)