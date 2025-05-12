import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import griddata

def load_wavefield_data(file, sensor_csv, verbose=False):
    """
    Load wavefield data from NPZ or pickle file and sensor locations from CSV.
    
    Parameters:
    -----------
    file : str
        Path to the NPZ or pickle file containing wavefield data
    sensor_csv : str
        Path to the CSV file containing sensor locations
        
    Returns:
    --------
    tuple: (wavefield, sensor_lat, sensor_lon, sensors_df)
    """
    # Load the npz file and extract the numpy array
    if file.suffix == '.npz':
        with np.load(file) as data:
            if len(data.files) == 1:
                wavefield = data[data.files[0]]
                wavefield = wavefield.T
            else:
                raise KeyError(f"Could not find 'instaseis' key in {file}. Available keys: {data.files}")
    elif file.suffix == '.pkl':
        with open(file, 'rb') as f:
            wavefield = pickle.load(f)
            wavefield = wavefield.T
    else:
        raise ValueError(f"Unsupported file format: {file.suffix}. Only .npz and .pkl files are supported.")

    if verbose:
        print(f"Loaded array shape: {wavefield.shape}")

    # Extract sensor locations
    sensors_df = pd.read_csv(sensor_csv)
    sensor_lat = sensors_df['latitude'].values
    sensor_lon = sensors_df['longitude'].values
    if verbose:
        print(f"Sensors df shape: {sensors_df.shape}")
    
    return wavefield, sensor_lat, sensor_lon, sensors_df

def compute_color_scale(wavefield, verbose=False):
    """
    Compute global color scale using percentile-based scaling.
    
    Parameters:
    -----------
    wavefield : np.ndarray
        The wavefield data array
        
    Returns:
    --------
    tuple: (global_min, global_max)
    """
    # Use percentile-based color scaling for better contrast
    p5, p95 = np.percentile(wavefield, [5, 95])
    global_min = p5
    global_max = p95
    if verbose:
        print(f"Color scale range (5th-95th percentile): {global_min:.3f} to {global_max:.3f}")
    return global_min, global_max

def convert_sensors_to_cartesian(sensor_lat, sensor_lon):
    """
    Convert sensor locations from lat/lon to Cartesian coordinates.
    
    Parameters:
    -----------
    sensor_lat : np.ndarray
        Sensor latitudes in degrees
    sensor_lon : np.ndarray
        Sensor longitudes in degrees
        
    Returns:
    --------
    tuple: (sensor_x, sensor_y, sensor_z)
    """
    sensor_x = np.cos(np.radians(sensor_lat)) * np.cos(np.radians(sensor_lon))
    sensor_y = np.cos(np.radians(sensor_lat)) * np.sin(np.radians(sensor_lon))
    sensor_z = np.sin(np.radians(sensor_lat))
    return sensor_x, sensor_y, sensor_z

def create_spherical_grid(n_theta=300, n_phi=300):
    """
    Create a high-resolution spherical grid for surface visualization.
    
    Parameters:
    -----------
    n_theta : int
        Longitude resolution
    n_phi : int
        Latitude resolution
        
    Returns:
    --------
    tuple: (theta_grid, phi_grid, x_grid, y_grid, z_grid)
    """
    # Include endpoint for theta to close the sphere (0 to 2π inclusive)
    theta = np.linspace(0, 2*np.pi, n_theta)  # longitude (0 to 2π)
    
    # Create phi distribution with more points near poles
    # φ = 0 is North Pole, φ = π/2 is Equator, φ = π is South Pole
    # We want high density near φ = 0 and φ = π (both poles)
    uniform_phi = np.linspace(0, 1, n_phi)
    
    # Simple approach: use quadratic mapping that concentrates at both ends
    # Map uniform [0,1] to a distribution concentrated at 0 and 1
    u_centered = 2 * uniform_phi - 1  # Map to [-1, 1]
    u_concentrated = np.sign(u_centered) * (np.abs(u_centered) ** 0.7)  # Concentrate at ±1
    u_back = (u_concentrated + 1) / 2  # Map back to [0, 1]
    phi = np.pi * u_back
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Convert grid to Cartesian coordinates
    x_grid = np.cos(theta_grid) * np.sin(phi_grid)
    y_grid = np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = np.cos(phi_grid)
    
    return theta_grid, phi_grid, x_grid, y_grid, z_grid

def setup_interpolation_points(sensor_x, sensor_y, sensor_z):
    """
    Set up sensor points for interpolation with longitude wraparound handling.
    
    Parameters:
    -----------
    sensor_x, sensor_y, sensor_z : np.ndarray
        Sensor Cartesian coordinates
        
    Returns:
    --------
    tuple: (sensor_points, sensor_theta, sensor_phi)
    """
    # Convert to spherical coordinates for interpolation
    sensor_theta = np.arctan2(sensor_y, sensor_x)  # longitude of sensors
    sensor_phi = np.arccos(np.clip(sensor_z, -1, 1))  # colatitude of sensors
    
    # Handle longitude wraparound by adding duplicate points at boundaries
    sensor_points = []
    for i, (theta_s, phi_s) in enumerate(zip(sensor_theta, sensor_phi)):
        # Add original point
        sensor_points.append([theta_s, phi_s])
        
        # Add wrapped points for longitude continuity at 0/2π boundary
        # Add point at theta + 2π (for negative theta values)
        sensor_points.append([theta_s + 2*np.pi, phi_s])
        
        # Add point at theta - 2π (for positive theta values)
        sensor_points.append([theta_s - 2*np.pi, phi_s])
    
    sensor_points = np.array(sensor_points)
    return sensor_points, sensor_theta, sensor_phi

def interpolate_to_grid(sensor_points, values, theta_grid, phi_grid, verbose=False):
    """
    Interpolate sensor values to a regular spherical grid.
    
    Parameters:
    -----------
    sensor_points : np.ndarray
        Sensor points in spherical coordinates with wraparound
    values : np.ndarray
        Sensor values to interpolate
    theta_grid, phi_grid : np.ndarray
        Grid coordinates for interpolation
        
    Returns:
    --------
    np.ndarray: Interpolated values reshaped to grid shape
    """
    # Create sensor values with wraparound
    sensor_values = []
    for val in values:
        sensor_values.extend([val, val, val])  # Original + 2 wrapped copies
    sensor_values = np.array(sensor_values)
    
    # Create grid points for interpolation
    grid_points = np.column_stack([theta_grid.ravel(), phi_grid.ravel()])
    
    # Interpolate values to grid using linear method for stability
    try:
        interpolated_values = griddata(
            sensor_points, sensor_values, grid_points, 
            method='linear', fill_value=np.nan
        )
        
        # Handle NaN values by using nearest neighbor interpolation as fallback
        nan_mask = np.isnan(interpolated_values)
        if np.any(nan_mask):
            if verbose:
                print(f"Filling {np.sum(nan_mask)} NaN values with nearest neighbor interpolation")
            interpolated_values[nan_mask] = griddata(
                sensor_points, sensor_values, grid_points[nan_mask], 
                method='nearest'
            )
    except Exception as e:
        if verbose:
            print(f"Linear interpolation failed, using nearest neighbor: {e}")
        interpolated_values = griddata(
            sensor_points, sensor_values, grid_points, 
            method='nearest'
        )
    
    # Reshape interpolated values to grid shape
    color_grid = interpolated_values.reshape(theta_grid.shape)
    return color_grid

def create_scatter_trace(sensor_x, sensor_y, sensor_z, values, sensor_lon, sensor_lat,
                        global_min, global_max, name=None, marker_size=6):
    """
    Create a Plotly 3D scatter trace with consistent styling.
    
    Parameters:
    -----------
    sensor_x, sensor_y, sensor_z : np.ndarray
        Sensor coordinates in Cartesian space
    values : np.ndarray
        Amplitude values for the sensors
    sensor_lon, sensor_lat : np.ndarray
        Sensor coordinates in geographic space (for hover info)
    global_min, global_max : float
        Color scale limits
    name : str, optional
        Name for the trace
    marker_size : int, optional
        Size of the markers (default: 6)
        
    Returns:
    --------
    go.Scatter3d: Plotly scatter trace
    """
    return go.Scatter3d(
        x=sensor_x,
        y=sensor_y,
        z=sensor_z,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=values,
            colorscale='RdBu_r',  # Reversed RdBu for consistency with surface plots
            cmin=global_min,
            cmax=global_max,
            colorbar=dict(
                title=dict(
                    text="Amplitude",
                    side="right"
                ),
                tickmode="linear",
                tick0=global_min,
                dtick=(global_max - global_min) / 8,
            ),
            showscale=True,
            opacity=1.0,
            line=dict(
                width=0.5,
                color='black'
            )
        ),
        hovertemplate='Lon: %{customdata[0]:.1f}°<br>Lat: %{customdata[1]:.1f}°<br>Amplitude: %{marker.color:.3f}<extra></extra>',
        customdata=np.column_stack([sensor_lon, sensor_lat]),
        name=name
    )

def create_surface_trace(x_grid, y_grid, z_grid, color_grid, theta_grid, phi_grid, 
                        global_min, global_max, name=None):
    """
    Create a Plotly surface trace with consistent styling.
    
    Parameters:
    -----------
    x_grid, y_grid, z_grid : np.ndarray
        Grid coordinates in Cartesian space
    color_grid : np.ndarray
        Color values for the surface
    theta_grid, phi_grid : np.ndarray
        Grid coordinates in spherical space (for hover info)
    global_min, global_max : float
        Color scale limits
    name : str, optional
        Name for the trace
        
    Returns:
    --------
    go.Surface: Plotly surface trace
    """
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=color_grid,
        colorscale='RdBu_r',  # Reversed RdBu for better contrast
        cmin=global_min,
        cmax=global_max,
        colorbar=dict(
            title=dict(
                text="Amplitude",
                side="right"
            ),
            tickmode="linear",
            tick0=global_min,
            dtick=(global_max - global_min) / 8,
        ),
        showscale=True,
        opacity=1.0,  # Full opacity for solid appearance
        hovertemplate='Lon: %{customdata[0]:.1f}°<br>Lat: %{customdata[1]:.1f}°<br>Amplitude: %{surfacecolor:.3f}<extra></extra>',
        customdata=np.stack([
            np.degrees(theta_grid), 
            90 - np.degrees(phi_grid)  # Convert colatitude to latitude
        ], axis=-1),
        name=name
    )

def get_common_layout_settings():
    """
    Get common layout settings for both visualization functions.
    
    Returns:
    --------
    dict: Common layout settings
    """
    return dict(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=1.5)
            ),
            bgcolor='white',
            xaxis=dict(
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='lightgray',
                range=[-1.2, 1.2]
            ),
            yaxis=dict(
                backgroundcolor='white',
                gridcolor='lightgray', 
                showbackground=True,
                zerolinecolor='lightgray',
                range=[-1.2, 1.2]
            ),
            zaxis=dict(
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='lightgray',
                range=[-1.2, 1.2]
            )
        )
    )

def visualize_wavefield_plotly(npz_file, sensor_csv, index, verbose=False):
    """
    Visualize wavefield from a compressed numpy .npz file as a solid, closed sphere surface.
    This version fixes the transparency and color issues.
    """
    # Load data using helper function
    wavefield, sensor_lat, sensor_lon, sensors_df = load_wavefield_data(npz_file, sensor_csv, verbose=verbose)

    # Get values for the specific time index
    values = wavefield[:, index]
    if verbose:
        print(f"Value range: {np.min(values):.3f} to {np.max(values):.3f}")
    
    # Compute color scale using helper function
    global_min, global_max = compute_color_scale(wavefield, verbose=verbose)
    
    # Convert sensor locations to Cartesian coordinates using helper function
    sensor_x, sensor_y, sensor_z = convert_sensors_to_cartesian(sensor_lat, sensor_lon)
    
    # Create spherical grid using helper function
    theta_grid, phi_grid, x_grid, y_grid, z_grid = create_spherical_grid()
    
    # Set up interpolation points using helper function
    sensor_points, sensor_theta, sensor_phi = setup_interpolation_points(sensor_x, sensor_y, sensor_z)
    
    # Interpolate to grid using helper function
    color_grid = interpolate_to_grid(sensor_points, values, theta_grid, phi_grid, verbose=verbose)
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Add the sphere surface using helper function
    surface_trace = create_surface_trace(
        x_grid, y_grid, z_grid, color_grid, theta_grid, phi_grid, 
        global_min, global_max
    )
    fig.add_trace(surface_trace)
    
    # Get common layout settings
    layout_settings = get_common_layout_settings()
    
    # Update layout with specific settings for this function
    layout_settings.update({
        'title': dict(
            text=f'Seismic Wavefield at Time Step {index}',
            x=0.5,
            font=dict(size=16)
        ),
        'width': 900,
        'height': 700,
        'margin': dict(l=0, r=0, b=0, t=40)
    })
    
    fig.update_layout(layout_settings)
    
    fig.show()
    
def visualize_wavefield_plotly_with_slider(npz_file, sensor_csv, n_frames=20, start_index=0, end_index=None, verbose=False):
    """
    Visualize wavefield with interactive time slider using Plotly's built-in slider.
    Pre-computes multiple time steps for smooth playback.
    
    Parameters:
    -----------
    npz_file : str
        Filename of the CTF X*train.npz
    sensor_csv : str  
        Filename of the sensor locations CSV
    n_frames : int
        Number of time steps to pre-compute for the slider
    start_index : int
        Starting time index
    end_index : int
        Ending time index (if None, uses last time step)
    """
    # Load data using helper function
    wavefield, sensor_lat, sensor_lon, sensors_df = load_wavefield_data(npz_file, sensor_csv, verbose=verbose)

    # Determine time indices for frames
    n_time = wavefield.shape[1]
    if end_index is None:
        end_index = n_time - 1
    
    frame_indices = np.linspace(start_index, end_index, n_frames, dtype=int)
    if verbose:
        print(f"Creating {n_frames} frames from index {start_index} to {end_index}")
    
    # Compute color scale using helper function
    global_min, global_max = compute_color_scale(wavefield, verbose=verbose)
    
    # Convert sensor locations to Cartesian coordinates using helper function
    sensor_x, sensor_y, sensor_z = convert_sensors_to_cartesian(sensor_lat, sensor_lon)
    
    # Create spherical grid using helper function
    theta_grid, phi_grid, x_grid, y_grid, z_grid = create_spherical_grid()
    
    # Set up interpolation points using helper function
    sensor_points, sensor_theta, sensor_phi = setup_interpolation_points(sensor_x, sensor_y, sensor_z)
    
    # Create the figure
    fig = go.Figure()
    
    # Pre-compute all frames
    if verbose:
        print("Pre-computing frames...")
    frames = []
    
    for i, time_idx in enumerate(frame_indices):
        if verbose:
            print(f"Computing frame {i+1}/{n_frames} (time index {time_idx})")
        
        # Get values for this time step
        values = wavefield[:, time_idx]
        
        # Interpolate to grid using helper function
        color_grid = interpolate_to_grid(sensor_points, values, theta_grid, phi_grid, verbose=verbose)
        
        # Create frame data using helper function
        frame_data = create_surface_trace(
            x_grid, y_grid, z_grid, color_grid, theta_grid, phi_grid, 
            global_min, global_max, name=f'Time {time_idx}'
        )
        
        frames.append(go.Frame(
            data=[frame_data],
            name=str(time_idx)
        ))
    
    # Add initial frame to figure
    fig.add_trace(frames[0].data[0])
    
    # Add frames to figure
    fig.frames = frames
    
    # Create slider steps
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time Index:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    # Create slider steps
    for i, time_idx in enumerate(frame_indices):
        slider_step = {
            "args": [
                [str(time_idx)],{
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300}
                }
            ],
            "label": str(time_idx),
            "method": "animate"
        }
        sliders_dict["steps"].append(slider_step)
    
    # Add play/pause buttons
    updatemenus = [{
        "buttons": [{
                "args": [None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
    
    # Get common layout settings and update with specific settings for this function
    layout_settings = get_common_layout_settings()
    layout_settings.update({
        'title': dict(
            text='Interactive Seismic Wavefield Visualization',
            x=0.5,
            font=dict(size=16)
        ),
        'width': 1000,
        'height': 800,
        'margin': dict(l=0, r=0, b=100, t=100),
        'sliders': [sliders_dict],
        'updatemenus': updatemenus
    })
    
    # Update layout
    fig.update_layout(layout_settings)
    
    if verbose:
        print("Visualization ready! Use the slider to change time steps or click Play for animation.")
    fig.show()

def plot_sensor_time_series(npz_file, sensor_csv, index, sensor_idx=None, verbose=False):
    """
    Plot time series for a specific sensor with current time step highlighted.
    
    Parameters:
    -----------
    npz_file : str
        Path to the NPZ file containing wavefield data
    sensor_csv : str
        Path to the CSV file containing sensor locations
    index : int
        Current time step to highlight
    sensor_idx : int, optional
        Index of sensor to plot. If None, uses middle sensor.
        
    Returns:
    --------
    None
    """
    wavefield, sensor_lat, sensor_lon, sensors_df = load_wavefield_data(npz_file, sensor_csv, verbose=verbose)

    # Use middle sensor if not specified
    if sensor_idx is None:
        sensor_idx = len(wavefield) // 2
    
    # Create time series plot
    fig_ts = go.Figure()
    
    fig_ts.add_trace(go.Scatter(
        x=list(range(len(wavefield[sensor_idx]))),
        y=wavefield[sensor_idx],
        mode='lines',
        name='Time series',
        line=dict(color='blue', width=2)
    ))
    
    fig_ts.add_vline(
        x=index,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Time step {index}"
    )
    
    fig_ts.update_layout(
        title=f'Time series for sensor (idx {sensor_idx}) at lat={sensor_lat[sensor_idx]:.2f}°, lon={sensor_lon[sensor_idx]:.2f}°',
        xaxis_title='Time step',
        yaxis_title='Amplitude',
        showlegend=True,
        width=800,
        height=400
    )
    
    fig_ts.show()

def visualize_wavefield_plotly_scatter(npz_file, sensor_csv, index, verbose=False):
    """
    Visualize wavefield from a compressed numpy .npz file as a 3D scatter plot.
    This version shows raw sensor data without interpolation.
    
    Parameters:
    -----------
    npz_file : str or Path
        Path to the NPZ or pickle file containing wavefield data
    sensor_csv : str or Path
        Path to the CSV file containing sensor locations
    index : int
        Time step index to visualize
    verbose : bool, optional
        Print debug information
        
    Returns:
    --------
    None
    """
    # Load data using helper function
    wavefield, sensor_lat, sensor_lon, sensors_df = load_wavefield_data(npz_file, sensor_csv, verbose=verbose)

    # Get values for the specific time index
    values = wavefield[:, index]
    if verbose:
        print(f"Value range: {np.min(values):.3f} to {np.max(values):.3f}")
    
    # Compute color scale using helper function
    global_min, global_max = compute_color_scale(wavefield, verbose=verbose)
    
    # Convert sensor locations to Cartesian coordinates using helper function
    sensor_x, sensor_y, sensor_z = convert_sensors_to_cartesian(sensor_lat, sensor_lon)
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Add scatter plot of sensors using helper function
    scatter_trace = create_scatter_trace(
        sensor_x, sensor_y, sensor_z, values, sensor_lon, sensor_lat,
        global_min, global_max, name=f'Sensors at time {index}'
    )
    fig.add_trace(scatter_trace)
    
    # Add a solid white sphere inside to hide back-side markers
    # Create a smaller sphere (radius ~0.95) to sit inside the unit sphere
    phi_sphere = np.linspace(0, np.pi, 30)
    theta_sphere = np.linspace(0, 2*np.pi, 60)
    phi_grid_sphere, theta_grid_sphere = np.meshgrid(phi_sphere, theta_sphere)
    
    # Convert to Cartesian coordinates for inner sphere
    radius = 0.95  # Slightly smaller than unit sphere
    x_sphere = radius * np.sin(phi_grid_sphere) * np.cos(theta_grid_sphere)
    y_sphere = radius * np.sin(phi_grid_sphere) * np.sin(theta_grid_sphere)
    z_sphere = radius * np.cos(phi_grid_sphere)
    
    # Add the inner white sphere
    inner_sphere = go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale=[[0, 'white'], [1, 'white']],  # Solid white
        showscale=False,
        opacity=1.0,
        hoverinfo='skip',  # No hover info for the sphere
        name='Background sphere',
        lighting=dict(
            ambient=1.0,    # Maximum ambient light (no shadows)
            diffuse=0.0,    # No diffuse lighting (eliminates shadows)
            specular=0.0,   # No specular highlights
            roughness=1.0,  # Maximum roughness (diffuses light)
            fresnel=0.0     # No fresnel effect
        ),
        lightposition=dict(
            x=0, y=0, z=0   # Light at origin (minimal shadow casting)
        )
    )
    fig.add_trace(inner_sphere)
    
    # Get common layout settings
    layout_settings = get_common_layout_settings()
    
    # Update layout with specific settings for scatter plot
    layout_settings.update({
        'title': dict(
            text=f'Seismic Wavefield (Scatter) at Time Step {index}',
            x=0.5,
            font=dict(size=16)
        ),
        'width': 900,
        'height': 700,
        'margin': dict(l=0, r=0, b=0, t=40)
    })
    
    # Update scene settings for better scatter visualization
    layout_settings['scene'].update({
        'camera': dict(
            eye=dict(x=2.5, y=2.5, z=1.8)  # Slightly further back for scatter points
        )
    })
    
    fig.update_layout(layout_settings)
    
    fig.show()

def visualize_wavefield_plotly_scatter_with_slider(npz_file, sensor_csv, n_frames=20, start_index=0, end_index=None, verbose=False):
    """
    Visualize wavefield with interactive time slider using 3D scatter plot.
    Pre-computes multiple time steps for smooth playback.
    
    Parameters:
    -----------
    npz_file : str or Path
        Path to the NPZ or pickle file containing wavefield data
    sensor_csv : str or Path
        Path to the CSV file containing sensor locations
    n_frames : int
        Number of time steps to pre-compute for the slider
    start_index : int
        Starting time index
    end_index : int
        Ending time index (if None, uses last time step)
    verbose : bool, optional
        Print debug information
        
    Returns:
    --------
    None
    """
    # Load data using helper function
    wavefield, sensor_lat, sensor_lon, sensors_df = load_wavefield_data(npz_file, sensor_csv, verbose=verbose)

    # Determine time indices for frames
    n_time = wavefield.shape[1]
    if end_index is None:
        end_index = n_time - 1
    
    frame_indices = np.linspace(start_index, end_index, n_frames, dtype=int)
    if verbose:
        print(f"Creating {n_frames} frames from index {start_index} to {end_index}")
    
    # Compute color scale using helper function
    global_min, global_max = compute_color_scale(wavefield, verbose=verbose)
    
    # Convert sensor locations to Cartesian coordinates using helper function
    sensor_x, sensor_y, sensor_z = convert_sensors_to_cartesian(sensor_lat, sensor_lon)
    
    # Create the figure
    fig = go.Figure()
    
    # Create inner sphere for background (same for all frames)
    phi_sphere = np.linspace(0, np.pi, 30)
    theta_sphere = np.linspace(0, 2*np.pi, 60)
    phi_grid_sphere, theta_grid_sphere = np.meshgrid(phi_sphere, theta_sphere)
    
    # Convert to Cartesian coordinates for inner sphere
    radius = 0.95  # Slightly smaller than unit sphere
    x_sphere = radius * np.sin(phi_grid_sphere) * np.cos(theta_grid_sphere)
    y_sphere = radius * np.sin(phi_grid_sphere) * np.sin(theta_grid_sphere)
    z_sphere = radius * np.cos(phi_grid_sphere)
    
    # Add the inner white sphere (static across all frames)
    inner_sphere = go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale=[[0, 'white'], [1, 'white']],  # Solid white
        showscale=False,
        opacity=1.0,
        hoverinfo='skip',  # No hover info for the sphere
        name='Background sphere',
        lighting=dict(
            ambient=1.0,    # Maximum ambient light (no shadows)
            diffuse=0.0,    # No diffuse lighting (eliminates shadows)
            specular=0.0,   # No specular highlights
            roughness=1.0,  # Maximum roughness (diffuses light)
            fresnel=0.0     # No fresnel effect
        ),
        lightposition=dict(
            x=0, y=0, z=0   # Light at origin (minimal shadow casting)
        )
    )
    
    # Pre-compute all frames
    if verbose:
        print("Pre-computing frames...")
    frames = []
    
    for i, time_idx in enumerate(frame_indices):
        if verbose:
            print(f"Computing frame {i+1}/{n_frames} (time index {time_idx})")
        
        # Get values for this time step
        values = wavefield[:, time_idx]
        
        # Create scatter trace for this frame using helper function
        scatter_trace = create_scatter_trace(
            sensor_x, sensor_y, sensor_z, values, sensor_lon, sensor_lat,
            global_min, global_max, name=f'Sensors at time {time_idx}'
        )
        
        # Create frame with both scatter and sphere
        frames.append(go.Frame(
            data=[scatter_trace, inner_sphere],
            name=str(time_idx)
        ))
    
    # Add initial frame to figure (first frame)
    fig.add_trace(frames[0].data[0])  # Scatter trace
    fig.add_trace(inner_sphere)       # Background sphere
    
    # Add frames to figure
    fig.frames = frames
    
    # Create slider steps
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time Index:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    # Create slider steps
    for i, time_idx in enumerate(frame_indices):
        slider_step = {
            "args": [
                [str(time_idx)],{
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300}
                }
            ],
            "label": str(time_idx),
            "method": "animate"
        }
        sliders_dict["steps"].append(slider_step)
    
    # Add play/pause buttons
    updatemenus = [{
        "buttons": [{
                "args": [None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
    
    # Get common layout settings and update with specific settings for this function
    layout_settings = get_common_layout_settings()
    layout_settings.update({
        'title': dict(
            text='Interactive Seismic Wavefield Scatter Visualization',
            x=0.5,
            font=dict(size=16)
        ),
        'width': 1000,
        'height': 800,
        'margin': dict(l=0, r=0, b=100, t=100),
        'sliders': [sliders_dict],
        'updatemenus': updatemenus
    })
    
    # Update scene settings for better scatter visualization
    layout_settings['scene'].update({
        'camera': dict(
            eye=dict(x=2.5, y=2.5, z=1.8)  # Slightly further back for scatter points
        ),
        # Uncomment to hide axes
        #'xaxis': dict(
            #visible=False,  # Hide entire x-axis
            #showbackground=False,
            #showgrid=False,
            #zeroline=False
        #),
        #'yaxis': dict(
            #visible=False,  # Hide entire y-axis
            #showbackground=False,
            #showgrid=False,
            #zeroline=False
        #),
        #'zaxis': dict(
            #visible=False,  # Hide entire z-axis
            #showbackground=False,
            #showgrid=False,
            #zeroline=False
        #)
    })
    
    # Update layout
    fig.update_layout(layout_settings)
    
    if verbose:
        print("Visualization ready! Use the slider to change time steps or click Play for animation.")
    fig.show()

def plot_wavefield_timeseries_heatmap(npz_file, sensor_csv, verbose=False):
    """
    Plot the full wavefield time series as a 2D heatmap.
    Shows time on x-axis, sensors on y-axis, and amplitude as color.
    
    Parameters:
    -----------
    npz_file : str or Path
        Path to the NPZ or pickle file containing wavefield data
    sensor_csv : str or Path
        Path to the CSV file containing sensor locations
    verbose : bool, optional
        Print debug information
        
    Returns:
    --------
    None
    """
    # Load data using helper function
    wavefield, sensor_lat, sensor_lon, sensors_df = load_wavefield_data(npz_file, sensor_csv, verbose=verbose)
    
    if verbose:
        print(f"Wavefield shape: {wavefield.shape}")
        print(f"Time steps: {wavefield.shape[1]}")
        print(f"Sensors: {wavefield.shape[0]}")
    
    # Compute color scale using helper function
    global_min, global_max = compute_color_scale(wavefield, verbose=verbose)
    
    # Create time and sensor indices for axis labels
    n_sensors, n_time = wavefield.shape
    time_indices = np.arange(n_time)
    sensor_indices = np.arange(n_sensors)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=wavefield,
        x=time_indices,
        y=sensor_indices,
        colorscale='RdBu_r',  # Same colorscale as other plots
        zmin=global_min,
        zmax=global_max,
        colorbar=dict(
            title=dict(
                text="Amplitude",
                side="right"
            ),
            tickmode="linear",
            tick0=global_min,
            dtick=(global_max - global_min) / 8,
        ),
        hoverongaps=False,
        hovertemplate='Time: %{x}<br>Sensor: %{y}<br>Amplitude: %{z:.3f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Seismic Wavefield Time Series',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Time Step',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Sensor Index',
            showgrid=True,
            gridcolor='lightgray'
        ),
        width=1000,
        height=600,
        margin=dict(l=60, r=60, b=60, t=80)
    )
    
    if verbose:
        print("Heatmap visualization ready!")
    fig.show()

def plot_wavefield_timeseries_heatmap_with_geography(npz_file, sensor_csv, sort_by='latitude', order='descending', verbose=False):
    """
    Plot the full wavefield time series as a 2D heatmap with sensors sorted by geographic location.
    Shows time on x-axis, sensors (sorted geographically) on y-axis, and amplitude as color.
    
    Parameters:
    -----------
    npz_file : str or Path
        Path to the NPZ or pickle file containing wavefield data
    sensor_csv : str or Path
        Path to the CSV file containing sensor locations
    sort_by : str, optional
        How to sort sensors: 'latitude', 'longitude', or 'distance_from_origin'
    order : str, optional
        How to sort sensors: 'ascending' or 'descending'
    verbose : bool, optional
        Print debug information
        
    Returns:
    --------
    None
    """
    # Load data using helper function
    wavefield, sensor_lat, sensor_lon, sensors_df = load_wavefield_data(npz_file, sensor_csv, verbose=verbose)
    
    if verbose:
        print(f"Wavefield shape: {wavefield.shape}")
        print(f"Time steps: {wavefield.shape[1]}")
        print(f"Sensors: {wavefield.shape[0]}")
    
    # Sort sensors by geographic criteria
    if sort_by == 'latitude':
        sort_indices = np.argsort(sensor_lat)
        sort_label = 'Latitude'
        sort_values = sensor_lat[sort_indices]
    elif sort_by == 'longitude':
        sort_indices = np.argsort(sensor_lon)
        sort_label = 'Longitude'
        sort_values = sensor_lon[sort_indices]
    elif sort_by == 'distance_from_origin':
        distances = np.sqrt(sensor_lat**2 + sensor_lon**2)
        sort_indices = np.argsort(distances)
        sort_label = 'Distance from Origin'
        sort_values = distances[sort_indices]
    else:
        raise ValueError("sort_by must be 'latitude', 'longitude', or 'distance_from_origin'")
    
    if order == 'descending':
        sort_indices = sort_indices[::-1]
    elif order == 'ascending':
        pass
    else:
        raise ValueError("order must be 'ascending' or 'descending'")
    
    # Reorder wavefield data according to sorting
    wavefield_sorted = wavefield[sort_indices, :]
    sensor_lat_sorted = sensor_lat[sort_indices]
    sensor_lon_sorted = sensor_lon[sort_indices]
    
    if verbose:
        print(f"Sensors sorted by {sort_by}")
        print(f"Range: {sort_values[0]:.2f} to {sort_values[-1]:.2f}")
    
    # Compute color scale using helper function
    global_min, global_max = compute_color_scale(wavefield, verbose=verbose)
    
    # Create time indices for x-axis
    n_sensors, n_time = wavefield.shape
    time_indices = np.arange(n_time)
    
    # Create custom hover text with geographic info
    hover_text = []
    for i in range(n_sensors):
        row = []
        for j in range(n_time):
            row.append(f'Time: {j}<br>Sensor: {sort_indices[i]}<br>Lat: {sensor_lat_sorted[i]:.2f}°<br>Lon: {sensor_lon_sorted[i]:.2f}°<br>Amplitude: {wavefield_sorted[i,j]:.3f}')
        hover_text.append(row)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=wavefield_sorted,
        x=time_indices,
        y=np.arange(n_sensors),
        colorscale='RdBu_r',  # Same colorscale as other plots
        zmin=global_min,
        zmax=global_max,
        colorbar=dict(
            title=dict(
                text="Amplitude",
                side="right"
            ),
            tickmode="linear",
            tick0=global_min,
            dtick=(global_max - global_min) / 8,
        ),
        hoverongaps=False,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Create custom y-axis labels showing geographic values at key positions
    n_ticks = min(10, n_sensors)  # Show up to 10 ticks
    tick_positions = np.linspace(0, n_sensors-1, n_ticks, dtype=int)
    tick_labels = [f'{sort_values[i]:.1f}°' for i in tick_positions]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Seismic Wavefield Time Series (sorted by {sort_by})',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Time Step',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=f'Sensors (sorted by {sort_label})',
            showgrid=True,
            gridcolor='lightgray',
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels
        ),
        width=1000,
        height=600,
        margin=dict(l=80, r=60, b=60, t=80)
    )
    
    if verbose:
        print(f"Geographic heatmap visualization ready! Sensors sorted by {sort_by}")
    fig.show()
