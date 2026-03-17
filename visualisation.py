import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import numpy as np
from skyfield.api import load, EarthSatellite
from datetime import datetime, timedelta

# Load satellite trajectory data from CSV
def plot_satellite_trajectory(csv_file, tle_lines=None):
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Check for required columns
    required_columns = {'x', 'y', 'z'}
    if not required_columns.issubset(df.columns):
        print(f"CSV must contain the following columns: {required_columns}")
        return

    # Create 3D plot for ECI coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['x'], df['y'], df['z'], label='Satellite Trajectory (ECI)')

    # Plot orbit from TLE if provided
    if tle_lines and len(tle_lines) == 3:
        ts = load.timescale()
        satellite = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
        t0 = datetime.utcnow()
        times = ts.utc(t0.year, t0.month, t0.day, t0.hour, t0.minute ,range(0, 6000, 30))
        eci_positions = satellite.at(times).position.m
        ax.plot(eci_positions[0], eci_positions[1], eci_positions[2], 'r--', label='TLE Orbit')

    # Plot a sphere to represent Earth
    earth_radius = 6371000
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Satellite Trajectory in ECI Coordinates')
    ax.legend()
    plt.grid(True)

    # Set equal aspect ratio
    max_range = np.array([df['x'].max()-df['x'].min(), df['y'].max()-df['y'].min(), df['z'].max()-df['z'].min()]).max() / 2.0
    mid_x = (df['x'].max()+df['x'].min()) * 0.5
    mid_y = (df['y'].max()+df['y'].min()) * 0.5
    mid_z = (df['z'].max()+df['z'].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

# Example usage (replace with actual values):
tle =["FLYING LAPTOP",
"1 42831U 17042G   25125.19906600  .00003089  00000+0  26197-3 0  9995",
"2 42831  97.4029 294.5720 0012808  74.9361 285.3279 14.98577215425308"]
plot_satellite_trajectory('data/out.csv', tle_lines=tle)
