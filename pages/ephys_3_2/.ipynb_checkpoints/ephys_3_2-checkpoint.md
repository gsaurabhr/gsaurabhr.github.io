# Df with different parameters
There is a discrepancy in the relative ordering of differentiation across layers in my and Will's analysis. We found that we had slight variations in our parameters, such as the window size for binning (and also a gaussian averaging that Will applies on top), sampling rate after binning and the units selected for analysis (I was rejecting units with SNR <= 2.5). Below, I play with these three parameters and see how they affect differentiation.

## Original (SNR >=2.5, sampling 50Hz, bin 50 ms)
![png](all_sessions_sampling_20_win_50_snr_2.5.png)

## All units (SNR >=0.0, sampling 50Hz, bin 50 ms)
![png](all_sessions_sampling_20_win_50_snr_0.0.png)

## Closest to Will (SNR >=0, sampling 10Hz, bin 20 ms)
![png](all_sessions_sampling_10_win_20_snr_0.0.png)

# Df for averaged responses
Analysis for two sessions is shown for comparison.

## Original (SNR >=2.5, sampling 50Hz, bin 50 ms)
![png](session_737581020_sampling_20_win_50_snr_2.5.png)
![png](session_757216464_sampling_20_win_50_snr_2.5.png)

## All units (SNR >=0.0, sampling 50Hz, bin 50 ms)
![png](session_737581020_sampling_20_win_50_snr_0.0.png)
![png](session_757216464_sampling_20_win_50_snr_0.0.png)
