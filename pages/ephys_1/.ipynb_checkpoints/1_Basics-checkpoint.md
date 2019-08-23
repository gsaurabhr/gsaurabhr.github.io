

```python
%load_ext autoreload
%autoreload 2
from ephys_utilities import *
```


```python
cache.get_sessions().head(n=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_type</th>
      <th>specimen_id</th>
      <th>genotype</th>
      <th>gender</th>
      <th>age_in_days</th>
      <th>project_code</th>
      <th>probe_count</th>
      <th>channel_count</th>
      <th>unit_count</th>
      <th>has_nwb</th>
      <th>structure_acronyms</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>737581020</th>
      <td>brain_observatory_1.1</td>
      <td>718643567</td>
      <td>wt</td>
      <td>M</td>
      <td>108.0</td>
      <td>NeuropixelVisualCoding</td>
      <td>6</td>
      <td>396</td>
      <td>601</td>
      <td>True</td>
      <td>[CA, DG, MB, TH, VISl, VISmma, VISp, VISpm, VI...</td>
    </tr>
    <tr>
      <th>739448407</th>
      <td>brain_observatory_1.1</td>
      <td>716813543</td>
      <td>wt</td>
      <td>M</td>
      <td>112.0</td>
      <td>NeuropixelVisualCoding</td>
      <td>6</td>
      <td>422</td>
      <td>654</td>
      <td>True</td>
      <td>[CA, DG, MB, TH, VIS, VISam, VISl, VISp, VISrl...</td>
    </tr>
  </tbody>
</table>
</div>




```python
SESSION = 737581020
session = cache.get_session_data(SESSION)
stimuli = session.stimulus_presentations
MAXTIMEMS = int(stimuli['stop_time'].values[-1])*1000
with suppress_stdout():
    stimulus_blocks = session.get_stimulus_epochs()
```

    /home/saurabh.gandhi/.local/lib/python3.6/site-packages/pandas/core/ops/__init__.py:1115: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      result = method(y)



```python
session.summarize()
```

    session 737581020 acquired on 2018-09-25 14:03:59-07:00
    # channels : 2218
    # probes   : 6
    # units    : 601
    # stimuli  : 70390
    Structures :  ['VISpm (13)', 'VISl (25)', 'VISmma (17)', 'TH (236)', 'CA (115)', 'VISp (40)', 'VISrl (69)', 'DG (41)', 'MB (45)']
    Stim names : ['spontaneous_activity', 'gabor_20_deg_250ms', 'flash_250ms', 'drifting_gratings', 'natural_movie_3', 'natural_movie_1', 'static_gratings', 'Natural Images']
    
    



![png](output_3_1.png)


---


```python
SAMPLING = 20
WINDOW = 100
```


```python
firing_rates = session.get_firing_rate(sampling=SAMPLING, window=WINDOW)
unit_ids = firing_rates.columns.values
```

    100%|██████████| 175/175 [00:45<00:00,  4.06it/s]


# Selective firing in some units


```python
# example firing rates
session.plot_firing_rate(unit_ids[3], (0, 4500))
session.plot_firing_rate(unit_ids[2], (0, 6250))
```


![png](output_8_0.png)



![png](output_8_1.png)



```python
plt.figure(figsize=(16, 2))
firing_rates[unit_ids[3]].plot(c=cm.Greys(0.5, 0.9), xlim=(1575, 1625))
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
plt.title('Zooming in...');
```


![png](output_9_0.png)


---


```python
# group units by area
groups = {}
for struc in set(session.good_units['structure_acronym']):
    groups[struc] = session.good_units[session.good_units['structure_acronym'] == struc].index.values
areas = {
    'all' : ['VISmma', 'CA', 'VISl', 'VISpm', 'VISp', 'MB', 'DG', 'VISrl', 'TH'],
    'all visual' : ['VISp', 'VISpm', 'VISl', 'VISmma', 'VISrl'],
    'all hippocampal' : ['DG', 'CA']
}
```

---


```python
mfr = {}
sfr = {}
block_times = {}
```


```python
# average response to drifting gratings
mfrd, sfrd, block_timesd = {}, {}, {}
idx = session.stimulus_conditions[session.stimulus_conditions['stimulus_name'] == 'drifting_gratings'].index.values
for stim in idx:
    block = stimuli[stimuli['stimulus_condition_id'] == stim]
    start_times = block['start_time'].values
    end_times = block['stop_time'].values
    durations = block['duration'].values
    duration = durations.min()
    fr = np.zeros((len(durations), len(unit_ids), 1+int(duration*1000/SAMPLING)))
    block_timesd[stim] = []
    for i in range(len(start_times)):
        t0 = start_times[i]+(1-(start_times[i]*1000/SAMPLING-int(start_times[i]*1000/SAMPLING)))*SAMPLING/1000
        block_timesd[stim].append(np.arange(t0, t0+duration, SAMPLING/1000))
        fr[i] = firing_rates.reindex(block_timesd[stim][-1], method='ffill').values.T
    mfrd[stim] = pd.DataFrame(data=fr.mean(axis=0).T,
                              columns=unit_ids,
                              index=np.arange(duration*1000/SAMPLING)*SAMPLING/1000)
    sfrd[stim] = pd.DataFrame(data=fr.std(axis=0).T,
                              columns=unit_ids,
                              index=np.arange(duration*1000/SAMPLING)*SAMPLING/1000)

mfr['drifting_gratings'] = pd.DataFrame(columns=unit_ids)
sfr['drifting_gratings'] = pd.DataFrame(columns=unit_ids)
block_times['drifting_gratings'] = []
for stim in sorted(mfrd.keys()):
    mfr['drifting_gratings'] = mfr['drifting_gratings'].append(mfrd[stim], ignore_index=True)
    sfr['drifting_gratings'] = sfr['drifting_gratings'].append(sfrd[stim], ignore_index=True)
    for i in range(len(block_timesd[stim])):
        if i == len(block_times['drifting_gratings']):
            block_times['drifting_gratings'].append([])
        block_times['drifting_gratings'][i] += list(block_timesd[stim][i])

for i in range(len(block_times['drifting_gratings'])):
    block_times['drifting_gratings'][i] = np.array(block_times['drifting_gratings'][i])
mfr['drifting_gratings'].index = np.arange(len(mfr['drifting_gratings'].index))*SAMPLING/1000
sfr['drifting_gratings'].index = np.arange(len(sfr['drifting_gratings'].index))*SAMPLING/1000
```


```python
# average response to movies
for stim in ['natural_movie_1', 'natural_movie_3']:
    block = stimulus_blocks[stimulus_blocks['stimulus_name'] == stim]
    start_times = block['start_time'].values
    end_times = block['stop_time'].values
    durations = block['duration'].values
    duration = durations.min()
    fr = np.zeros((len(durations), len(unit_ids), 1+int(duration*1000/SAMPLING)))
    block_times[stim] = []
    for i in range(len(start_times)):
        t0 = start_times[i]+(1-(start_times[i]*1000/SAMPLING-int(start_times[i]*1000/SAMPLING)))*SAMPLING/1000
        block_times[stim].append(np.arange(t0, t0+duration, SAMPLING/1000))
        fr[i] = firing_rates.reindex(block_times[stim][-1], method='ffill').values.T
    mfr[stim] = pd.DataFrame(data=fr.mean(axis=0).T,
                             columns=unit_ids,
                             index=np.arange(duration*1000/SAMPLING)*SAMPLING/1000)
    sfr[stim] = pd.DataFrame(data=fr.std(axis=0).T,
                             columns=unit_ids,
                             index=np.arange(duration*1000/SAMPLING)*SAMPLING/1000)
```

# Mean response to stimuli

1. Big difference in firing rates of the same neuron for repeated stimulus
2. The variance is much larger than mean, indicating that the differnce is more than just poison noise
3. The large variance persists after averaging the firing rate over up to 200 ms windows, indicating that it is not simply a matter of temporal offsets

Questions

1. Finding the correct timescale to look at responses (average over 10 ms, or 100 ms?)

## Response to natural movie 1


```python
stim = 'natural_movie_1'
area = 'VISp'
for i in range(3):
    try:
        unit_id = session.get_units_by_area(area)[i]
        plt.figure(figsize=(16, 2))
        plt.plot(mfr[stim].index.values,
                 firing_rates.reindex(block_times[stim][0], method='ffill')[unit_id].values,
                 c=cm.Blues(0.5, 0.4))
        plt.plot(mfr[stim].index.values,
                 firing_rates.reindex(block_times[stim][1], method='ffill')[unit_id].values,
                 c=cm.Greens(0.5, 0.4))
        mfr[stim][unit_id].plot(c=cm.Reds(0.5, 0.4))
#         plt.twinx()
#         ((sfr[stim]*sfr[stim])/mfr[stim])[unit_id].plot(c='k', lw=0.5)
#         plt.ylim(0, 10)
        plt.annotate('Mean var/mena = %.0f'%((sfr[stim]*sfr[stim])/mfr[stim])[unit_id],
                     (0.8, 0.8), xycoords='axes fraction')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')
    except:
        pass
```


![png](output_18_0.png)



![png](output_18_1.png)



![png](output_18_2.png)


## Response to drifting gratings


```python
stim = 'drifting_gratings'
area = 'VISp'
for i in range(3):
    try:
        unit_id = session.get_units_by_area(area)[i]
        plt.figure(figsize=(16, 2))
        plt.plot(mfr[stim].index.values,
                 firing_rates.reindex(block_times[stim][0], method='ffill')[unit_id].values,
                 c=cm.Blues(0.5, 0.4))
        plt.plot(mfr[stim].index.values,
                 firing_rates.reindex(block_times[stim][1], method='ffill')[unit_id].values,
                 c=cm.Greens(0.5, 0.4))
        mfr[stim][unit_id].plot(c=cm.Reds(0.5, 0.4))
#         plt.twinx()
#         ((sfr[stim]*sfr[stim])/mfr[stim])[unit_id].plot(c='k', lw=0.5)
#         plt.ylim(0, 10)
        plt.annotate('Mean var/mena = %.0f'%((sfr[stim]*sfr[stim])/mfr[stim])[unit_id],
                     (0.8, 0.8), xycoords='axes fraction')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')
    except:
        pass
```


![png](output_20_0.png)



![png](output_20_1.png)



![png](output_20_2.png)


---

# A bit of CNNs
Correlation between SD and ND decreases systematically with depth for three different networks

### VGG16
![png](vgg16.png)

### InceptionV3
![png](inceptionV3.png)

### ResNet50
![png](resnet50.png)


```python
# import dill
# #dill.dump_session('notebook_env.db')
# dill.load_session('../../../OneDrive - Allen Institute/notebook_env.db')
```


```python

```
