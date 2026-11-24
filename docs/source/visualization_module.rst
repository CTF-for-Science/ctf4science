Visualization Module
====================

This guide describes how to use the ``Visualization`` module within the CTF for Science framework to generate, customize, and save plots that help assess and understand model performance.

Overview
--------

The ``Visualization`` module provides tools to:

* Plot predicted vs. ground truth **trajectories**
* Compare **error metrics** across time or sub-datasets
* Visualize **histograms** of variable distributions
* Display **power spectral densities (PSDs)** for spatiotemporal data
* Perform **side-by-side 2D comparisons** of predictions vs. truth
* Plot **spatial averages** and **2D contours** for each field of multi-field datasets
* Automate plot generation for entire batch runs

These plots are saved under each run's ``visualizations/`` directory inside the ``results/`` folder.

Usage
-----

Import and Initialize
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ctf4science.visualization_module import Visualization
   viz = Visualization()  # Uses default config

To use a custom configuration:

.. code-block:: python

   viz = Visualization(config_path='path/to/your/visualization_config.yaml')

Plotting Functions
------------------

Please refer to :doc:`api` for the full API of the visualization module.

Customization
-------------

Visualization appearance is governed by a config file (``default_visualization_config.yaml``). You can override settings like:

* ``figure_size``
* Line ``colors`` and ``linestyles``
* Image ``colormap``

Example override:

.. code-block:: python

   fig = viz.plot_trajectories(truth, [predictions], figure_size=(12, 8), colors={'truth': 'black'})

Multi-field Datasets
--------------------

Some datasets (e.g. ``msfr``, ``mhd-2cyls``, ``dynasty``) store several physical fields side by side along the column axis. The ``spatial_averages`` and ``contour2d`` plots draw one panel per field, using these keys from the dataset YAML:

* ``evaluation_params.num_fields``: number of fields. Without ``field_sizes``, the columns are split into ``num_fields`` equal parts.
* ``metadata.field_sizes`` (optional): width of each field, in column order, for fields of unequal width. Must sum to the number of columns.
* ``metadata.fields`` (optional): field names, used as panel labels instead of ``Field 1``, ``Field 2``, etc.

For example, ``dynasty`` has a 1-column mass flow rate followed by 122 temperatures:

.. code-block:: yaml

   evaluation_params:
     num_fields: 2
   metadata:
     fields: [mass_flow_rate, temperatures]
     field_sizes: [1, 122]

Output Locations
----------------

Plots are saved in:

::

   results/<dataset>/<model>/<batch_id>/pair<id>/visualizations/

Each run will contain the visualizations that are defined for the dataset in the dataset.yaml file

* ``trajectories.png``
* ``histograms.png``
* ``psd.png``
* ``errors.png``
* ``2d_comparison.png``

Tips
----

* Use ``generate_all_plots()`` to automate full visualization.
* Make sure ``predictions.npy`` and ``evaluation_results.yaml`` exist before calling batch plotting.
* If using 2D comparisons, your data must be 2D arrays.
* Use Jupyter to call plotting functions interactively and tweak parameters.
