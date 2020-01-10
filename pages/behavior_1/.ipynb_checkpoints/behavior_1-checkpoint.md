# MFR differentiation, PCIst and mean firing rate during task performance
---

## Example sessions form single mouse

### mfr differentiation
![mfr_diff](mfr_diff_comparisons_single_session.png)

### PCIst
PCIst compares the number of 'state transitions' in the baseline period (before stimulus) to the response period (after stimulus). Here I use the image presentation as the stimulus. Images are presented every 750ms, and the presentation lasts for 250ms (so that there is 500ms of blank screen between presentations). The baseline and response windows are chosen to be 300ms, so that the baseline is sufficiently after the prefious trial, and the response is long enough to capture any latish responses. It turns out that even in the original application of PCIst to EEG data, they used 300ms windows.
![PCIst](PCIst_comparisons_single_session.png)

### mean firing rate
![mean firing rate comparisons](mean_firing_comparisons_single_session.png)

## Aggregate across all sessions
This is not straightforward because the overall patterns of differentiation in different areas is not conserved across sessions. Thus, we might be interested in these overall patterns, or we might be more interested in the modulation wrt different responses that is on top of this overall pattern.

Here, I choose to eliminate the overall pattern and focus on the modulation caused by different kinds of responses. To do this, differentiation values in each area are normalized by the mean differentiation value for no_change condition in that area.

### mfr differentiation
![mfr_diff (all sessions)](mfr_diff_comparisons_all_sessions.png)

### PCIst
![PCIst (all sessions)](PCIst_comparisons_all_sessions.png)

### mean firing rate
![mean firing rate comparisons (all sessions)](mean_firing_comparisons_all_sessions.png)

## More figures

### variation of differentiation across images
Overall, differentiation (or PCIst) does not depend strongly on the image itself. Below, the differentiation is plotted for 'no_change' conditions for different images, to visualize the variation across images.

![mfr_diff across images](mfr_diff_variation_images.png)