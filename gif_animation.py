import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy as sp
import scipy.ndimage

XT = 10
YT = 10
XR = 10
YR = 10

# Import data
r = np.load('r.npy')

# print(r[0])


fig = plt.figure()
ax = plt.axes(xlim=(0, XR - 1), ylim=(0, YR - 1))
ax.set(xticklabels=[])
ax.set(yticklabels=[])
line, = ax.plot([], [])


def init():
    line.set_data([], [])
    return line,

def calculate_COM(s):
    '''
    Center of Mass Function

    Description
        - This function calculates the 'centre of mass' of the stregnths of
        synaptic connection between each tectal neuron and the retinal sheet
        - For each tectal neuron the row of 's' that coressponds to the
        strengths of synaptic connections to the neurons of the retinal sheet is
        reshaped into a np.array of dimensions 'YR' by 'XR'
        - The centre of mass of the synapse strengths is calculated as
        coordinates
        - These coordinates are then stored seperately in different matrices
        ('COM_X' and 'COM_Y')

    Parameteres
        s - numpy.array containing the stregnth of the synaptic connections
            between each retinal neuron and every tectal neuron
        XT - integer x dimension of the tectal sheet (number of neurons)
        YT - integer y dimension of the tectal sheet (number of neurons)
        XR - integer x dimension of the retinal sheet (number of neurons)
        YR - integer y dimension of the retinal sheet (number of neurons)

    Returns: 'COM_X, COM_Y' - two numpy.arrays respectively containing the X and
    Y coordinates
    '''
    # Create zeros arrays for the COM X and Y values
    s = np.array([list(t) for t in s])

    COM_Y = np.zeros((YT * XT))
    COM_X = np.zeros((YT * XT))

    for tectal_neuron in range(XT * YT):
        COM_grid = np.reshape(s[tectal_neuron, :], (YR, XR))
        # Calculate COM for 'tectal_neuron'
        COM = sp.ndimage.measurements.center_of_mass(COM_grid)
        # Store X and Y coordinates of COM in two seperate arrays
        # Populate 'COM_X' and 'COM_Y'
        COM_Y[tectal_neuron] = COM[0]
        COM_X[tectal_neuron] = COM[1]

    return COM_X, COM_Y


# Plot
def plot_fish_net(t):
    '''
    Plot function

    Description
        - The linear output from the 'calculate_COM()' function is reshaped
        into an np.array with dimensions ('YT' by 'XT')
            -> this facilitates the joining up of the data points to form the
                net
        - The fishnet plot is then plotted using the COM data generated
        - The plot is then saved

    Parameters
        repeat_count - integer corresponding to which repeat of the run()
                        function resulted in the image saved
        s - numpy.array containing the stregnth of the synaptic connections
            between each retinal neuron and every tectal neuron
        XT - integer x dimension of the tectal sheet (number of neurons)
        YT - integer y dimension of the tectal sheet (number of neurons)
        XR - integer x dimension of the retinal sheet (number of neurons)
        YR - integer y dimension of the retinal sheet (number of neurons)
        COM_X - np.array of the 'x' coordinates of the tectal neurons' centres
                of mass
        COM_Y - np.array of the 'y' coordinates of the tectal neurons' centres
                of mass
    '''
    print(t)
    COM_X, COM_Y = calculate_COM(r[t])

    COM_X_grid = np.reshape(COM_X, (YT, XT))
    COM_Y_grid = np.reshape(COM_Y, (YT, XT))

    ax.clear()

    for i in range(XT):
        ax.plot(COM_X_grid[i, :], COM_Y_grid[i, :], color='b')

    for j in range(YT):
        ax.plot(COM_X_grid[:, j], COM_Y_grid[:, j], color='b')

    ax.set_xlim(0, XR - 1)
    ax.set_ylim(0, YR - 1)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])

    ax.set_xlabel('%.2f %%' % (t * 0.8))
    return line, ax


# Call the animator
anim = animation.FuncAnimation(fig, plot_fish_net, init_func=init,
                               frames=125, interval=40, blit=False)


anim.save('animation.gif', fps=25, writer='imagemagick')

# plt.show()
