# Seismic Dataset

This dataset is generated using the Python library Instaseis, which produces synthetic seismograms from Green's function databases. These seismograms record waveforms of ground motion in response to an earthquake at a given source location. Instaseis can generate data in terms of displacement, velocity, or acceleration. In this dataset, we use velocity, which has units of m/s.

## Stations (sensors)

Stations (the virtual seismometers) are assumed to be on the Earth's surface. Their locations are recorded as x-y-z or latitude-longitude fields on a generic sphere with a 6371km radius. For each earthquake, each station records a 3-component waveform: Z (vertical), N (north-south horizontal), and E (east-west horizontal). We only take the Z component for simplicity.

## Sources

Earthquake events are generated with random values of location, magnitude, and other characteristics such as type of earthquake (normal-thrust vs strike-slip).

## The data

For each earthquake, each station (sensor) records 1 time series of velocities in the Z (vertical) direction. A typical training set has the shape of (2000, 2048), corresponding to 2000 time-steps and 2048 sensors.

## Data normalisation

Waveform time series could differ by orders of magnitude due to 1. the log scale of earthquake magnitudes and 2. the reduction of wave amplitudes due to the distance between the station and the source. Within the time series, there are also large variations in the amplitude.

To mitigate this, we normalise waveform traces to unit variance and zero mean. The benefit is that neural network models can focus on waveform shape. The drawback of normalisation is that we lose the physical meaning of amplitude (e.g. difference between nearby and far stations, or between a magnitude 4 and magnitude 7 earthquake), but this is acceptable because the waveform shapes are the key characteristic of the time series that we want the models to learn and predict. 

The normalisation is done for each individual time series, since, for instance, X1train and X1test need to be normalised by a common mean and variance to ensure continuity and predictability. We normalise time series per group, using the mean and variance of the whole group. The grouping is as follows:

- X1train, X1test
- X2train, X2test, X3test
- X3train, X4test, X5test
- X4train, X6test
- X5train, X7test
- X6train
- X7train
- X8train
- X9train, X8test
- X10train, X9test
