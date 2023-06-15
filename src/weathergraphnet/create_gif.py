import matplotlib.animation as animation #type: ignore
import matplotlib.pyplot as plt #type: ignore
import xarray as xr

# Define the path to the NetCDF file
nc_file='prog_vars_-30.0_4000.0_1000.0_DOM01_ML_-30.0_4000.0_1000.0_20080801T000000Z.nc'
nc_path='/scratch/sadamov/icon/icon-nwp/cpu/experiments/atmcirc-straka_93/' + nc_file

# Define the variable to plot
var_name = 'temp'

# Open the NetCDF file
ds = xr.open_dataset(nc_path)

var = ds[var_name]
# Calculate the minimum and maximum values of the variable
var_min = var.min()
var_max = var.max()

# Create a new figure object
fig, ax = plt.subplots()

# Plot the first time step of the variable
im = var.isel(time=0).plot(vmin=var_min, vmax=var_max, ax=ax)
plt.gca().invert_yaxis()  # type: ignore # invert the y-axis
plt.title('Time: 0 seconds')

def update(frame):
    # Update the data of the current plot
    time_in_seconds = round((var.time[frame] - var.time[0]).item() * 24 * 3600)
    im.set_array(var.isel(time=frame))
    plt.title(f'Time: {time_in_seconds:.0f} seconds')
    return im

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=var.shape[0],
    interval=100, 
    blit=False
)

# Define the filename for the output gif
output_filename = 'animation.gif' + nc_file

# Save the animation as a gif
ani.save(output_filename, writer='imagemagick', dpi=100)

