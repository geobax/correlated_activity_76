'''
George Baxter
Neural Activity Model of Retinotopic Map Formation
Cython Script
'''

import numpy as np
import scipy as sp
import math
cimport numpy as np
from cpython cimport bool
cimport cython
import time

# Functions

# Initialise Synaptic Connections
@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2, mode='c'] initialise_synaptic_connections(double ND_mean, double ND_sd, int XT, int YT, int XR, int YR):
	'''
	Initialisation function

	Description
		- Function produces a numpy.array ('s') containing the initial stregnth 
		of the synaptic connections between each retinal neuron and every tectal
		neuron
		- ('XT' * 'YT') tectal neurons are the y dimension of 's'
		- ('XR' * 'YR') retinal neurons are the x dimension of 's'
		- Values are randomly sampled from a normal disribution of mean 
		'ND_mean', and standard deviation 'ND_sd'

	Parameters
		ND_mean - float mean of the normal distribution used to determine the 
				values in 's'
		ND_sd - float standard deviation of the aforementioned normal 
				distribution
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: 's' - numpy.ndarray containing the initial stregnth of the synaptic 
	connections between each retinal neuron and every tectal neuron
	'''
	return np.random.normal(ND_mean, ND_sd, (XT * YT, XR * YR))


# Configure Retinal Polarity Markers
@cython.wraparound(False)
@cython.boundscheck(False)
def configure_retinal_polarity_markers(int XR, int YR, bool default_polarity_markers):
	'''
	Configure retinal polarity markers

	Description
		- Model recquires a square of 4 adjacent retinal neurons to have 
		stronger initial synaptic connections to a square of 4 adjacent tectal 
		neurons. This ensures the tectal receptive fields form in the correct 
		orientation
		- If 'default_polarity_markers' is 'True' then the retinal polarity 
		markers are set to be the centre 4 retinal neurons (if the X or Y axes 
		are odd numbers then the upper, or left 'centre' option is used)
			-> this ensures nucleation begins at the centre of the map
		- If 'default_polarity_markers' is 'False' this function randomly 
		selects a square of 4 retinal neurons to serve as the retinal polarity 
		markers
		- First polarity marker is identified by randomly selecting x and y 
		coordinates within the retinal grid (dimensions: 'YR' x 'XR')
		- Other 3 polarity markers are identified by selecting adjacent x and y 
		coordinates
		- x and y coordinates of polarity markers are then converted into an 
		integer corresponding to the selected retinal neuron's column in 's'

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
		default_polarity_markers - boole, if 'True' PMs default to the top left 
									4 neurons on the retinal sheet. if 'False' 
									then polarity markers are selected at random

	Returns: 'R_PM1', 'R_PM2', 'R_PM3', 'R_PM4' - integers identifying the 
	columns of 's' that correspond to the selected retinal polarity markers
	'''
	if default_polarity_markers:
		'''Integer division of 'XR' and 'YR' gives the top-left neuron
		coordinate. The other PMs are derived from the surrounding right and 
		bottom 3 neurons'''
		R_PM_x1 = XR//2
		R_PM_y1 = YR//2
		R_PM_x2 = R_PM_x1 + 1
		R_PM_y2 = R_PM_y1
		R_PM_x3 = R_PM_x1
		R_PM_y3 = R_PM_y1 + 1
		R_PM_x4 = R_PM_x2
		R_PM_y4 = R_PM_y3

	else:
		# Generate coordinates in the retinal grid: 'R_PM_y' is the Y coordinate and 'R_PM_x' the X coordinate
		# Coordinates for retinal polarity marker 1 ('R_PM1'). This will be the top left retinal polarity marker (hence ranges up to 'YR-1' and 'XR-1')
		R_PM_y1 = np.random.randint(YR-1)
		R_PM_x1 = np.random.randint(XR-1)
		# Generate coordinates ('R_PM_y2', 'R_PM_x2') for retinal polarity marker 2 ('R_PM2') which is in the same row as 'R_PM1' (top right)
		# Generate 'R_PM_y2' -> same as 'R_PM_y1'
		R_PM_y2 = R_PM_y1
		# Generate 'R_PM_x2'
		R_PM_x2 = R_PM_x1+1
		# Generate coordinates for 'R_PM3' and 'R_PM4' which lie directly below 'R_PM1' and 'R_PM2'
		# Generate 'y' coordinates
		R_PM_y3 = R_PM_y1-1
		R_PM_y4 = R_PM_y2-1
		# Generate 'x' coordinates
		R_PM_x3 = R_PM_x1
		R_PM_x4 = R_PM_x2

	# Convert 'y' and 'x' coordinates for the R_PMs to correspond to specific retinal neurons in 's'
	R_PM1 = R_PM_y1 * XR + R_PM_x1
	R_PM2 = R_PM_y2 * XR + R_PM_x2
	R_PM3 = R_PM_y3 * XR + R_PM_x3
	R_PM4 = R_PM_y4 * XR + R_PM_x4

	return R_PM1, R_PM2, R_PM3, R_PM4

# Configure Tectal Polarity Markers
@cython.wraparound(False)
@cython.boundscheck(False)
def configure_tectal_polarity_markers(int XT, int YT, bool default_polarity_markers):
	'''
	Configure tectal polarity markers

	Description
		- Model recquires a square of 4 adjacent retinal neurons to have 
		stronger initial synaptic connections to a square of 4 adjacent tectal 
		neurons. This ensures the tectal receptive fields form in the correct 
		orientation
		- If 'default_polarity_markers' is 'True' then the tectal polarity 
		markers are set to be the top left 4 tectal neurons
		- If 'default_polarity_markers' is 'True' then the tectal polarity 
		markers are set to be the centre 4 tectal neurons (if the X or Y axes 
		are odd numbers then the upper, or left 'centre' option is used)
			-> this ensures nucleation begins at the centre of the map
		- First polarity marker is identified by randomly selecting x and y 
		coordinates within the tectal grid (dimensions: 'YT' x 'XT')
		- Other 3 polarity markers are identified by selecting adjacent x and y 
		coordinates
		- x and y coordinates of polarity markers are then converted into an 
		integer corresponding to the selected tectal neuron's row in 's'

	Parameters
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		default_polarity_markers - boole, if 'True' PMs default to the top left 
									4 neurons on the tectal sheet. if 'False' 
									then polarity markers are selected at random

	Returns: 'T_PM1', 'T_PM2', 'T_PM3', 'T_PM4' - integers identifying the 
	rows of 's' that correspond to the selected tectal polarity markers
	'''
	if default_polarity_markers:
		'''Integer division of 'XT' and 'YT' gives the top-left neuron
		coordinate. The other PMs are derived from the surrounding right and 
		bottom 3 neurons'''
		T_PM_x1 = XT//2
		T_PM_y1 = YT//2
		T_PM_x2 = T_PM_x1 + 1
		T_PM_y2 = T_PM_y1
		T_PM_x3 = T_PM_x1
		T_PM_y3 = T_PM_y1 + 1
		T_PM_x4 = T_PM_x2
		T_PM_y4 = T_PM_y3

	else:	
		# Generate coordinates in the tectal grid: 'T_PM_y' is the Y coordinate and 'T_PM_x' the X coordinate
		# Coordinates for tectal polarity marker 1 ('T_PM1'). This will be the top left tectal polarity marker (hence ranges up to 'YT-1' and 'XT-1')
		T_PM_y1 = np.random.randint(YT-1)
		T_PM_x1 = np.random.randint(XT-1)
		# Generate coordinates ('T_PM_y2', 'T_PM_x2') for tectal polarity marker 2 ('T_PM2') which is in the same row as 'T_PM1' (top right)
		# Generate 'T_PM_y2' -> same as 'T_PM_y1'
		T_PM_y2 = T_PM_y1
		# Generate 'T_PM_x2'
		T_PM_x2 = T_PM_x1+1
		# Generate coordinates for 'T_PM3' and 'T_PM4' which lie directly below 'T_PM1' and 'T_PM2'
		# Generate 'y' coordinates
		T_PM_y3 = T_PM_y1-1
		T_PM_y4 = T_PM_y2-1
		# Generate 'x' coordinates
		T_PM_x3 = T_PM_x1
		T_PM_x4 = T_PM_x2

	# Convert 'y' and 'x' coordinates for the T_PMs to correspond to specific retinal neurons in 's'
	T_PM1 = T_PM_y1 * XT + T_PM_x1
	T_PM2 = T_PM_y2 * XT + T_PM_x2
	T_PM3 = T_PM_y3 * XT + T_PM_x3
	T_PM4 = T_PM_y4 * XT + T_PM_x4

	return T_PM1, T_PM2, T_PM3, T_PM4

# Increase Polarity Marker Synapse Strengths
@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2, mode='c'] increase_polarity_marker_synapse_strengths(np.ndarray[double, ndim=2, mode='c'] s, double PM_strength_increase, 
																								int R_PM1, int R_PM2, int R_PM3, int R_PM4, int T_PM1, 
																								int T_PM2, int T_PM3, int T_PM4):
	'''
	Increase Polarity Marker Synapse Strengths

	Description
		- Function increases the strength of the synapses (values in 's') 
		between the corresponding retinal and tectal polarity markers by a 
		magnitude of 'PM_strength_increase' (e.g. the synapse between 'R_PM1' 
		and 'T_PM1')

	Parameters
		s - numpy.ndarray (dimensions: [XT*YT, XR*YR]) of the strengths of the 
			synaptic connections between retinal neurons (X axis) and tectal 
			neurons (Y axis)
		PM_strength_increase - float coefficient for the increase in stregnth of
							the synapses between retinal and tectal polarity
							markers
		R_PM1 - integer defining the column of 's' that R_PM1 corresponds to
		R_PM2 - integer defining the column of 's' that R_PM2 corresponds to
		R_PM3 - integer defining the column of 's' that R_PM3 corresponds to
		R_PM4 - integer defining the column of 's' that R_PM4 corresponds to
		T_PM1 - integer defining the row of 's' that T_PM1 corresponds to
		T_PM2 - integer defining the row of 's' that T_PM2 corresponds to
		T_PM3 - integer defining the row of 's' that T_PM3 corresponds to
		T_PM4 - integer defining the row of 's' that T_PM4 corresponds to

	Returns: 's' - numpy.ndarray containing the initial stregnth of the synaptic 
	connections between each retinal neuron and every tectal neuron
	'''
	# Increase the strength of synapse between the corresponding retinal and tectal polarity markers
	s[T_PM1, R_PM1] *= PM_strength_increase
	s[T_PM2, R_PM2] *= PM_strength_increase
	s[T_PM3, R_PM3] *= PM_strength_increase
	s[T_PM4, R_PM4] *= PM_strength_increase

	return s

# Implement Square Polarity Markers
@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2, mode='c'] square_polarity_markers(np.ndarray[double, ndim=2, mode='c'] s, 
																	bool default_polarity_markers, int XR, int YR, 
																	int XT, int YT, double PM_strength_increase):
	'''
	Implement Square Polarity Markers

	Description
		- Function implements a 2x2 grid of polarity markers between the retinal
		and tectal sheets in order to ensure that the orientation of the two
		sheets is matched during development of the retinotopic maps. This style
		of polarity markers is in accordance with those suggested in the Wilshaw 
		& von der Malsburg 1986
		- If 'default_polarity_markers' is True then the PMs default to the top
		left corner of both the retinal and tectal sheets
		- If 'default_polarity_markers' is False then the PMs are selected at
		random in both sheets

	Parameters
		s - numpy.array (dimensions: [XT*YT, XR*YR]) of the strengths of the 
			synaptic connections between retinal neurons (X axis) and tectal 
			neurons (Y axis)
		default_polarity_markers - boole, if 'True' PMs default to the top left 
									4 neurons on the tectal sheet. if 'False' 
									then polarity markers are selected at random
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
		PM_strength_increase - float coefficient for the increase in stregnth of
							the synapses between retinal and tectal polarity
							markers
	Returns: 's' - numpy.ndarray containing the stregnth of the synaptic 
	connections between each retinal neuron and every tectal neuron
	'''
	cdef int R_PM1, R_PM2, R_PM3, R_PM4, T_PM1, T_PM2, T_PM3, T_PM4

	R_PM1, R_PM2, R_PM3, R_PM4 = configure_retinal_polarity_markers(XR, YR, default_polarity_markers)
	T_PM1, T_PM2, T_PM3, T_PM4 = configure_tectal_polarity_markers(XT, YT, default_polarity_markers)
	s = increase_polarity_marker_synapse_strengths(s, PM_strength_increase, R_PM1, R_PM2, R_PM3, R_PM4, T_PM1, T_PM2, T_PM3, T_PM4)

	return s

# Implement Graded Polarity Markers
@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2, mode='c'] graded_polarity_markers(np.ndarray[double, ndim=2, mode='c'] s, 
																	int XT, int YT, int XR, int YR):
	'''
	Implement Graded Polarity Markers

	Description
		- Function increases the strength of the synapses (values in 's') 
		between retinal and tectal neurons based on their relative positions in 
		the retinal and tectal sheets (normalised to their sizes)
		- y and x coordinates (termed 'a' and 'b' for the retinal sheet, and 
		'c' and 'd' for the tectal sheet) are derived from the neuron number (as
		indicated by the row and column of the synapse under consideration in 
		's')
		- These x and y coordinates are then normalised to the dimensions of the
		retinal/tectal sheet
		- The pythagorian distance (dist) between these normalised coordinates is 
		calculated, and normalised to the maximum possibe separation
		- Synapse stregnths are then increased in a graded way based on 'dist', 
		up to half the maximum separation

	Parameters
		s - numpy.ndarray (dimensions: [XT*YT, XR*YR]) of the strengths of the 
			synaptic connections between retinal neurons (X axis) and tectal 
			neurons (Y axis)
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: 's' - numpy.ndarray containing the stregnth of the synaptic 
	connections between each retinal neuron and every tectal neuron
	'''	
	cdef float INV_SQRT_2, a, b, c, d, dist_squared, dist

	# 1/sqrt(2) calculated for later use
	INV_SQRT_2 = 1.0 / math.sqrt(2)

	for i in range(XR*YR):
		for j in range(XT*YT):
			# For synapse 'j,i' in 's' calculate 'a,b' & 'c,d'
			# 'a' is the integer division of 'i' by 'XR'
			a = float(i//XR)
			# 'b' is the remainder division of 'i' by 'XR'
			b = float(i%XR)
			# 'c' is the integer division of 'j' by 'XT'
			c = float(j//XT)
			# 'd' is the remainder division of 'j' by 'XT'
			d = float(j%XT)

			''' Normalise 'a', 'b', 'c', 'd' by their respective dimension of 
			either the retinal or tectal sheet '''
			a = a/YR
			b = b/XR
			c = c/YT
			d = d/XT

			# Calculate 'dist_squared'
			dist_squared = (a-c)*(a-c) + (b-d)*(b-d)
			# Calculate 'dist'
			dist = math.sqrt(dist_squared)
			# Normalise 'dist' to the maximum possible value (sqrt(2))
			dist = dist/INV_SQRT_2

			# Update 's[j, i]' based on the calculated 'dist'
			if dist < 0.5:
				'''At distances below 0.5, synaptic stregnth is increased
				linearly between 1x and 5x increase in stregnth -> described by 
				equation: mulitplicative stregnth increase = -8*dist + 5'''
				s[j,i] = (-8*dist + 5) * s[j,i]

	return s

# Implement Polarity Markers
def implement_polarity_markers(PM_type, s, default_polarity_markers, XR, YR, XT, 
								YT, PM_strength_increase):
	'''
	Implement Polarity Markers

	Description
		- If 'PM_type' == 'Square' then the 'square_polarity_markers' function 
		is used to set up the PMs
		- if 'PM_type' == 'Graded' then the 'graded_polarity_markers' function is
		used to set up the PMs
		- if 'PM_type' == 'none' then 's' is returned unmodified

	Parameters
		PM_type - string, determines which of the polarity marker functions is 
				used
		s - numpy.array (dimensions: [XT*YT, XR*YR]) of the strengths of the 
			synaptic connections between retinal neurons (X axis) and tectal 
			neurons (Y axis)
		default_polarity_markers - boole, if 'True' PMs default to the top left 
									4 neurons on the tectal sheet. if 'False' 
									then polarity markers are selected at random
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
		PM_strength_increase - float coefficient for the increase in stregnth of
							the synapses between retinal and tectal polarity
							markers
	Returns: 's' - numpy.array containing the stregnth of the synaptic 
	connections between each retinal neuron and every tectal neuron
	'''
	if PM_type == "square":
		return square_polarity_markers(s, default_polarity_markers, XR, YR, XT, 
										YT, PM_strength_increase)
	elif PM_type == "graded":
		return graded_polarity_markers(s, XT, YT, XR, YR)

	elif PM_type == "none":
		return s

# Normalisation
@cython.wraparound(True)
@cython.boundscheck(True)
cdef np.ndarray[double, ndim=2, mode='c'] normalise(np.ndarray[double, ndim=2, mode='c'] s, double ND_mean, int XT, int YT, int XR, int YR):
	'''
	Normalisation function

	Description
		- Mean of each row of 's' is calculated. This gives the mean synaptic
		weighting of the connections between a tectal neuron and all of the
		retinal neurons
		- Each value in the aforementioned row is then normalised to the initial
		expected mean: 'ND_mean'

	Parameters
		s - numpy.ndarray (dimensions: [XT*YT, XR*YR]) of the stregnths of the 
			synaptic connections between retinal neurons (X axis) and tectal 
			neurons (Y axis)
		ND_mean - float mean value of the normal distribution used to set up the 
				initial synaptic stregnths in 's'
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: 's' - numpy.ndarray of the stregnths of retinal-tectal synaptic 
	stregnths, normalised to 'ND_mean'
	'''
	cdef int y, x
	cdef double mean_T

	for y in range(XT*YT):
		# Calculate mean value of retinal connections to tectal neuron 'y'
		mean_T = np.mean(s[y, :])
		for x in range(XR*YR):
			# Normalise each retinal connection ('x') to tectal neuron 'y' such that the average for all 'x' is 'ND_mean'
			s[y, x] = (ND_mean / mean_T) * s[y, x]
	return s

# Select Pairs of Retinal Neurons
@cython.boundscheck(False)
cdef np.ndarray retinal_neuron_pairs(int XR, int YR):
	'''
	Select Pairs of Retinal Neurons

	Description
		- Model recquires a pair of adjacent retinal neurons to be randomly 
		selected and activated
		- This function performs the above
		- First retinal neuron is identified by randomly selecting x and y 
		coordinates within the retinal grid (dimensions: 'YR' x 'XR')
		- Second retinal neuron is identified by selecting adjacent x and y 
		coordinates
		- x and y coordinates of selected retinal neurons are then converted 
		into an integer corresponding to the selected retinal neuron's column 
		in 's'

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: retinal_neurons - np.array containing 'RN1', 'RN2', integers
	identifying the columns of 's' that correspond to the selected retinal
	neurons
	'''
	cdef int x1, x2, x3, x4, y1, y2, y3, y4, RN1, RN2

	# Generate coordinates in the retinal grid: 'y' is the Y coordinate and 'x' the X coordinate
	# Coordinates of retinal neuron 1 ('RN1')
	y1 = np.random.randint(YR)
	x1 = np.random.randint(XR)
	# Generate coordinates ('y2', 'x2') for an adjacent retinal neuron ('RN2')
	# Generate 'y2'
	if y1 == 0:
		y2 = np.random.choice([y1, y1+1])
	elif y1 == YR-1:
		y2 = np.random.choice([y1, y1-1])
	else:
		y2 = np.random.choice([y1-1, y1, y1+1])
	# Generate 'x2'
	if y2 == y1:
		if x1 == 0:
			x2 = x1+1
		elif x1 == XR-1:
			x2 = x1-1
		else:
			x2 = np.random.choice([x1-1, x1+1])
	else:
		x2 = x1
	# Convert 'y' and 'x' coordinates for 'RN1' and 'RN2' to correspond to specific retinal neurons in 's'
	RN1 = y1 * XR + x1
	RN2 = y2 * XR + x2

	retinal_neurons = np.array([RN1, RN2])

	return retinal_neurons

# Select Squares of Retinal Neurons
@cython.boundscheck(False)
cdef np.ndarray retinal_neuron_squares(int XR, int YR):
	'''
	Select Squares of Retinal Neurons

	N.B. Increase 'theta' and 'epsilon' by 2x (relative to 
	original parameters in the paper)

	Description
		- This function selects and activates 4 adjacent retinal neurons in a 
		2x2 square
		- First retinal neuron is identified by randomly selecting x and y 
		coordinates within the retinal grid (dimensions: 'YR' x 'XR')
		- Second, third and fourth retinal neurons are identified by selecting 
		adjacent x and y coordinates
		- x and y coordinates of selected retinal neurons are then converted 
		into an integer corresponding to the selected retinal neuron's column of
		synapses in 's'

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: retinal_neurons - array containing 'RN1', 'RN2', 'RN3', 'RN4', 
	integers identifying the columns of 's' that correspond to the selected 
	retinal neurons
	'''

	cdef int x1, x2, x3, x4, y1, y2, y3, y4, RN1, RN2, RN3, RN4

	'''Generate coordinates in the retinal grid: 'y' is the Y coordinate and 'x'
	the X coordinate'''
	# Coordinates of retinal neuron 1 ('RN1')
	y1 = np.random.randint(YR)
	x1 = np.random.randint(XR)
	
	# Generate coordinates ('y2', 'x2') for an adjacent retinal neuron ('RN2')
	# Generate 'y2'
	if y1 == 0:
		y2 = 1
	elif y1 == YR-1:
		y2 = YR-2
	else:
		y2 = np.random.choice([y1-1, y1+1])
	# Generate 'x2'
	x2 = x1

	# Generate coordinates ('y3', 'x3') for an adjacent retinal neuron ('RN3')
	# Generate 'y3'
	y3 = y1
	# Generate 'x3'
	if x1 == 0:
		x3 = 1
	elif x1 == XR-1:
		x3 = XR-2
	else:
		x3 = np.random.choice([x1-1, x1+1])

	# Generate coordinates ('y4', 'x4') for an adjacent retinal neuron ('RN4')
	# Generate 'y4'
	y4 = y2
	x4 = x3

	'''Convert 'y' and 'x' coordinates for 'RN1' and 'RN2' to correspond to
	specific retinal neurons in 's' '''
	RN1 = y1 * XR + x1
	RN2 = y2 * XR + x2
	RN3 = y3 * XR + x3
	RN4 = y4 * XR + x4

	retinal_neurons = np.array([RN1, RN2, RN3, RN4])

	return retinal_neurons

# Select Retinal Neurons for Sweep Pattern
@cython.boundscheck(False)
cdef np.ndarray retinal_neuron_sweep(int XR, int YR, int loop_count):
	'''
	Select Rows/Columns of Retinal Neurons for Sweep Activation Pattern

	N.B. Increase 'theta' and 'epsilon' by 'XR'x (relative to 
	original parameters in the paper)

	Description
		- This function selects and activates either a row or column of the
		retinal sheet
		- Which row or column to activate is determined by the remainder
		division of the 'loop_count' by (XR+YR), e.g. the total number of rows
		and columns
		- The x and y coordinates of the retinal neurons in the activated 
		row/column are then stored in seperate arrays ('x_array' and 'y_array')
		respectively
		- x and y coordinates of selected retinal neurons are then converted 
		into an integer corresponding to the selected retinal neuron's column of
		synapses in 's'
			-> stored in np.array 'retinal_neurons'

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
		loop_count - integer number of times the retinal sheet has been
					activated

	Returns: 'retinal neurons' - np.array holding the integers identifying the
	columns of 's' that correspond to the selected retinal neurons
	'''

	cdef np.ndarray x_array, y_array

	# Calculate the sweep number ('sweep_num')
	sweep_num = loop_count%(YR+XR)

	if sweep_num < YR:
		''' For 'sweep_num' < YR function activates column 'sweep_num' of the retinal
		sheet '''
		# All x values are the same in this column
		x_array = np.ones(YR) * sweep_num
		# y values vary between 0 and 'YR'
		y_array = np.arange(YR)

		retinal_neurons = np.zeros(YR)
		for i in range(YR):
			retinal_neurons[i] = y_array[i] * XR + x_array[i]

	else:
		''' For 'sweep_num' >= YR function activates row ('sweep_num' - XR) of the
		retinal sheet '''
		# x values vary between 0 and 'XR'
		x_array = np.arange(XR)
		# All y values are the same in this row
		y_array = np.ones(XR) * (sweep_num - XR)

		retinal_neurons = np.zeros(XR)
		for i in range(XR):
			retinal_neurons[i] = y_array[i] * XR + x_array[i]

	return retinal_neurons

# Select Two Pairs of Retinal Neurons
@cython.boundscheck(False)
cdef np.ndarray retinal_neuron_2_pairs(int XR, int YR):
	'''
	Select Two Pairs of Retinal Neurons

	N.B. Increase 'theta' and 'epsilon' by 2x (relative to 
	original parameters in the paper)

	Description
		- This function randomly selects and activates two pairs of adjacent
		retinal neurons
		- First retinal neuron is identified by randomly selecting x and y 
		coordinates within the retinal grid (dimensions: 'YR' x 'XR')
		- Second retinal neuron is identified by selecting adjacent x and y 
		coordinates
		- The same process is repeated for the second pair
		- x and y coordinates of selected retinal neurons are then converted 
		into an integer corresponding to the selected retinal neuron's column 
		in 's'
			-> these are then stored in np.array 'retinal_neruons'

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: retinal_neurons - np.array containing 'RN1', 'RN2', integers
	identifying the columns of 's' that correspond to the selected retinal
	neurons
	'''

	cdef int x1, x2, x3, x4, y1, y2, y3, y4, RN1, RN2, RN3, RN4

	# Generate coordinates in the retinal grid: 'y' is the Y coordinate and 'x' the X coordinate
	
	# Pair 1
	# Coordinates of retinal neuron 1 ('RN1')
	y1 = np.random.randint(YR)
	x1 = np.random.randint(XR)
	# Generate coordinates ('y2', 'x2') for an adjacent retinal neuron ('RN2')
	# Generate 'y2'
	if y1 == 0:
		y2 = np.random.choice([y1, y1+1])
	elif y1 == YR-1:
		y2 = np.random.choice([y1, y1-1])
	else:
		y2 = np.random.choice([y1-1, y1, y1+1])
	# Generate 'x2'
	if y2 == y1:
		if x1 == 0:
			x2 = x1+1
		elif x1 == XR-1:
			x2 = x1-1
		else:
			x2 = np.random.choice([x1-1, x1+1])
	else:
		x2 = x1

	# Convert 'y' and 'x' coordinates for 'RN1' and 'RN2' to correspond to specific retinal neurons in 's'
	RN1 = y1 * XR + x1
	RN2 = y2 * XR + x2

	# Pair 2
	''' Coordinates of retinal neuron 3 ('RN3'). RN3 cannot be equal to either
		RN1 or RN2 '''
	RN3 = RN1
	while RN3 == RN1 or RN3 == RN2:
		y3 = np.random.randint(YR)
		x3 = np.random.randint(XR)

		# Calculate 'RN3'
		RN3 = y3 * XR + x3

	# Generate coordinates ('y4', 'x4') for an adjacent retinal neuron ('RN4')
	''' Coordinates of retinal neuron 4 ('RN4'). RN4 cannot be equal to either
		RN1 or RN2 '''
	RN4 = RN1
	while RN4 == RN1 or RN4 == RN2:
		# Generate 'y4'
		if y3 == 0:
			y4 = np.random.choice([y3, y3+1])
		elif y3 == YR-1:
			y4 = np.random.choice([y3, y3-1])
		else:
			y4 = np.random.choice([y3-1, y3, y3+1])
		# Generate 'x4'
		if y4 == y3:
			if x3 == 0:
				x4 = x3+1
			elif x3 == XR-1:
				x4 = x3-1
			else:
				x4 = np.random.choice([x3-1, x3+1])
		else:
			x4 = x3

		RN4 = y4 * XR + x4

	# Store 'RN1', 'RN2', 'RN3' and 'RN4' in 'retinal_neurons'
	retinal_neurons = np.array([RN1, RN2, RN3, RN4])

	return retinal_neurons

# Select Single Retinal Neurons
@cython.boundscheck(False)
cdef np.ndarray retinal_neuron_singles(int XR, int YR):
	'''
	Select Single Retinal Neurons

	N.B. This function is NOT expected to work
	N.B.2 Reduce 'theta' and 'epsilon' by 0.5x

	Description
		- This function performs selects single retinal neurons to be activated
		- The retinal neuron is identified by randomly selecting x and y 
		coordinates within the retinal grid (dimensions: 'YR' x 'XR')
		- x and y coordinates of selected retinal neuron are then converted 
		into an integer corresponding to the selected retinal neuron's column 
		in 's'
			-> these are then stored in np.array 'retinal_neruons'

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: retinal_neurons - np.array containing 'RN1', 'RN2', integers
	identifying the columns of 's' that correspond to the selected retinal
	neurons
	'''

	cdef int y1, x1, RN1, RN2

	# Generate coordinates in the retinal grid: 'y' is the Y coordinate and 'x' the X coordinate
	# Coordinates of retinal neuron 1 ('RN1')
	y1 = np.random.randint(YR)
	x1 = np.random.randint(XR)

	# Convert 'y' and 'x' coordinates for 'RN1' to correspond to specific retinal neurons in 's'
	RN1 = y1 * XR + x1

	retinal_neurons = np.array([RN1])

	return retinal_neurons

# Select Retinal Neurons for Strobe Pattern
cdef np.ndarray retinal_neuron_strobe(int XR, int YR):
	'''
	Select Retinal Neurons for Strobe Pattern

	N.B. Increase 'theta' and 'epsilon' by 'XR*YR/2'x and reduce 'h' (relative to 
	original parameters in the paper)

	Description
		- This function selects and activates the entire retinal sheet
		- np.array 'retinal_neurons' is created holding the integers
		corresponding to every retinal neuron

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
					
	Returns: 'retinal neurons' - np.array holding the integers identifying the
	columns of 's' that correspond to the selected retinal neurons
	'''

	retinal_neurons = np.arange(XR*YR)

	return retinal_neurons


# Select Two Single Retinal Neurons
cdef np.ndarray retinal_neuron_2_singles(int XR, int YR):
	'''
	Select 2 Non-Correlated Retinal Neurons

	Description
		- This function performs select two individual retinal neurons to be
		activated
		- The retinal neurons are identified by randomly selecting x and y 
		coordinates within the retinal grid (dimensions: 'YR' x 'XR')
		- x and y coordinates of selected retinal neurons are then converted 
		into integers corresponding to the selected retinal neurons' column 
		in 's'
			-> these are then stored in np.array 'retinal_neruons'

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)

	Returns: retinal_neurons - np.array containing 'RN1', 'RN2', integers
	identifying the columns of 's' that correspond to the selected retinal
	neurons
	'''

	cdef int y1, x1, RN1, RN2, y2, x2

	# Generate coordinates in the retinal grid: 'y' is the Y coordinate and 'x' the X coordinate
	# Coordinates of retinal neuron 1 ('RN1')
	y1 = np.random.randint(YR)
	x1 = np.random.randint(XR)

	# Convert 'y' and 'x' coordinates for 'RN1' to correspond to specific retinal neurons in 's'
	RN1 = y1 * XR + x1

	RN2 = RN1
	while RN2 == RN1:
		y2 = np.random.randint(YR)
		x2 = np.random.randint(XR)

		# Calculate 'RN2'
		RN2 = y2 * XR + x2

	retinal_neurons = np.array([RN1, RN2])

	return retinal_neurons

# Select Retinal Neurons for Occular Dominance
cdef np.ndarray retinal_neuron_ocular_dominance(int XR, int YR, int loop_count):
	'''
	Select Retinal Neurons for Ocular Dominance

	N.B. Increase 'theta' and 'epsilon' by '(XR*YR)/4'x (relative
	to original parameters in the paper)

	Description
		- This function alternatley selects and activates either the left or
		right half of the retinal sheet
		- 'y_array', np.array is created that holds the 'y' coordinates of the
		retinal neurons to be activated. This represents all of the rows of the
		retinal sheet (YR), repeated for half the columns (XR/2)
		- 'x_array', np.array is created that holds the 'x' coordinates of the
		retinal neurons to be activated
		- 'x_array' alternately represents either the left or right half of the
		retinal sheet (as determined by remainder division of 'loop_count' by 2)
		- np.array 'retinal_neurons' is created holding the integers
		corresponding to each retinal neuron activated

	Parameters
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
		loop_count - integer number of times the retinal sheet has been
					activated

	Returns: 'retinal neurons' - np.array holding the integers identifying the
	columns of 's' that correspond to the selected retinal neurons
	'''

	cdef np.ndarray y_array, x_array

	# Array of y values of the retinal neurons to be activated
	y_array = np.repeat(np.arange(YR), XR/2)

	# Determine whether the 'loop_count' is odd or even
	if loop_count%2 == 0:
		# If even, 'x_array' represents the left half of the retinal sheet
		x_array = np.tile(np.arange(XR/2), YR)

	else:
		# If odd, 'x_array' represents the right half of the retinal sheet
		x_array = np.tile(np.arange(XR/2, XR), YR)

	retinal_neurons = np.zeros(XR*YR/2)
	for i in range(XR*YR/2):
		retinal_neurons[i] = y_array[i] * XR + x_array[i]	

	return retinal_neurons

# Select Retinal Neuron Activation Pattern
cdef np.ndarray activate_retinal_neurons(str activity_pattern, int XR, 
											int YR, int loop_count):
	'''
	Select Retinal Neuron Activation Pattern

	Description
		- Based on the input string 'activity_pattern' this function determines
		which retinal activity pattern function is used

	Parameters
		activity_pattern - string determining which retinal activity pattern
							function is used
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
		loop_count - integer number of times the retinal sheet has been
					activated

	Returns: 'retinal neurons' - np.array holding the integers identifying the
	columns of 's' that correspond to the selected retinal neurons
	'''
	if activity_pattern == "pairs":
		return retinal_neuron_pairs(XR, YR)

	elif activity_pattern == "squares":
		return retinal_neuron_squares(XR, YR)

	elif activity_pattern == "sweep":
		return retinal_neuron_sweep(XR, YR, loop_count)

	elif activity_pattern == "2_pairs":
		return retinal_neuron_2_pairs(XR, YR)

	elif activity_pattern == "singles":
		return retinal_neuron_singles(XR, YR)

	elif activity_pattern == "strobe":
		return retinal_neuron_strobe(XR, YR)

	elif activity_pattern == "2_singles":
		return retinal_neuron_2_singles(XR, YR)

	elif activity_pattern == "ocular_dominance":
		return retinal_neuron_occular_dominance(XR, YR, loop_count)

# Activate Tectal Sheet
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline np.ndarray[double, ndim=2, mode='c'] activate_tectal_sheet(int XT, int YT, retinal_neurons, 
																		np.ndarray[double, ndim=2, mode='c'] s):
	'''
	Activate Tectal Sheet

	N.B. Increase 'theta' and 'epsilon' and reduce 'h' (relative to 
	original parameters in the paper)

	Description
		- The initial activity (before tectal-tectal interactions are 
		considered) of the tectal sheet is determined by: 1 * (strength of the 
		synaptic connections between the activated retinal neurons and each 
		tectal neuron)
		- This function creates a linear numpy.array ('H_linear') containing all
		of the tectal neurons' ('YT' * 'XT' neurons) activity, determined by the 
		sum of the connection strength between each of the activated retinal 
		neurons ('retinal_neurons') and the tectal neurons (as determined by the
		columns of 's')
		- 'H_linear' is then reshaped into numpy.array with the same dimensions 
		as the tectal sheet ('YT' x 'XT'). This matrix is named 'H_grid_initial'

	Parameters
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		retinal_neurons - np.ndarray containing 'RN1', 'RN2', integers
						identifying the columns of 's' that correspond to the
						selected retinal neurons
		s - numpy.array containing the stregnth of the synaptic connections 
			between each retinal neuron and every tectal neuron

	Returns: 'H_grid_initial' - numpy.array of the activity of the tectal 
	neurons
	'''
	cdef np.ndarray[double, ndim=1, mode='c'] H_linear
	cdef int i, j

	# Linear zeros matrix of tectal activity
	H_linear = np.zeros((YT * XT))
	# Update 'H_linear' from 's' based on 'retinal_neurons'
	for i in range(XT * YT):
		for j in retinal_neurons:
			H_linear[i] += s[i, j]
	# Reshape H_linear to be a grid (dimensions: 'YT' by 'XT')
	H_grid_initial = np.reshape(H_linear, (YT, XT))

	return H_grid_initial

# Threshold
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[double, ndim=2, mode='c'] threshold(np.ndarray[double, ndim=2, mode='c'] H_grid, double theta, int XT, int YT):
	'''
	Threshold function

	Description
		Input matrix values < threshold value (theta) = zero
		Input matrix values > threshold value (theta) = (value - theta)

	Parameters
		H_grid - numpy.ndarray of the activity of the tectal neurons
		theta - float threshold value
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)

	Returns: 'H_star' - numpy.ndarray of thresholded values
	'''
	cdef np.ndarray[double, ndim=2, mode='c'] H_star
	cdef int j, i

	# Create a copy of H_grid
	H_star = H_grid.copy()
	# Apply thresholding
	for j in range(YT):
		for i in range(XT):
			if H_star[j, i] > theta:
				H_star[j, i] = H_star[j, i] - theta
			else:
				H_star[j, i] = 0
	return H_star

# Convergence
def convergence(double mean_new_H_grid, double mean_H_grid):
	'''
	Convergence function

	Description
		- Function calculates the absolute difference between the mean activity 
		of the tectal neurons before and after a loop of the 
		'tectal_interactions()' function
		- If the mean activity has not changed by more than 0.5% then the tectal 
		sheet is considered to have converged

	Parameters
		mean_new_H_grid - float mean value of the tectal neuron activities after 
						a loop of 'tectal_interactions()'
		mean_H_grid - float mean value of the tectal neuron activities before 
					a loop of 'tectal_interactions()'

	Returns: 'converged' - boole, 'True' if the 'convergence' function has been 
	satisfied, 'False' if it has not
	'''
	return np.abs((mean_new_H_grid - mean_H_grid)) < (0.005 * mean_H_grid)


# Tectal Interactions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[double, ndim=2, mode='c'] tectal_interactions(np.ndarray[double, ndim=2, mode='c'] H_grid_initial, 
																	int XT, int YT, double dt, double beta, double gamma, double delta, 
																	double alpha, double theta, bool verbose):
	'''
	Tectal Interactions

	Description
		- Model states that tectal neurons with activities greater than the 
		threshold ('theta') have short range excitatory connections and long 
		range inhibitory connections to one another
		- Neurons 1 Manhattan distance away excite each other with magnitude: 
		('beta' * post-threshold potential)
		- Neurons 2 Manhattan distance away excite each other with magnitude: 
		('gamma' * post-threshold potential)
		- Neurons 3 Manhattan distance away inhibit each other with magnitude: 
		('delta' * post-threshold potential)
		- The neuronal membrane potential decays with decay constant 'alpha'
		- This function sets 'H_grid_initial' (the tectal activity based solely 
		on the activation of the randomly selected retinal neurons) as the first 
		iteration of 'H_grid'
		- The mean (excluding the boundary layers of zeros) is calculated. This 
		will be used when calculating convergence
		- 'dH_dt' for the tectal neurons is calculated using the above tectal 
		interactions
		- 'dH_dt' is then multiplied by a small time step ('dt') and 'H_grid' is 
		updated by ('dH_dt' * 'dt')
		- These interactions are then looped over until the tectal sheet 
		converges

	Parameters
		H_grid_initial - numpy.ndarray of the activity of the tectal 
						neurons, due only to the activation of the retinal 
						neurons (not tectal - tectal interactions)
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		dt - float time step for the integration of the tectal interactions
		beta - float co-efficient of the excitatory interactions between neurons 
				that are 1 Manhattan distance away
		gamme - float co-efficient of the excitatory interactions between 
				neurons that are 2 Manhattan distances away
		delta - float co-efficient of the inhibitory interactions between 
				neurons that are 3 Manhattan distances away
		alpha - float tectal membrane decay constant
		theta - float threshold value
		verbose - boole, if true prints 'mean_H_grid' and 'mean_new_H_grid', as 
				well as slowing the CPU for 1 second to allow time for printing 
				w/o CPU overload

	Returns: 'new_H_grid' - numpy.ndarray containing the activities of each tectal 
	neuron after convergence has been satisfied
	'''
	cdef np.ndarray[double, ndim=2, mode='c'] H_grid, dH_dt, H_star
	cdef int n_loops
	cdef double mean_H_grid, mean_new_H_grid

	# Define the tectal interactions within a np.ndarray. This will later be used to construct 'dH_dt'
	tectal_modifiers = np.array([[0., 0., 0., delta, 0., 0., 0.],
	                           [0., 0., delta, gamma, delta, 0., 0.],
	                       [0., delta, gamma, beta, gamma, delta, 0.],
	                       [delta, gamma, beta, 0., beta, gamma, delta],
	                       [0., delta, gamma, beta, gamma, delta, 0.],
	                           [0., 0., delta, gamma, delta, 0., 0.],
	                               [0., 0., 0., delta, 0., 0., 0.]])

	# Set 'H_grid_initial' as the first 'H_grid'
	H_grid = H_grid_initial.copy()
	# Create zeros matrix to hold dH/dts calculated below
	dH_dt = np.zeros((YT, XT))
	# Declare 'converged = False'
	converged = False
	# Iterate whilst 'converged' remains false
	n_loops = 0
	while not converged:

		# Apply thresholding
		H_star = threshold(H_grid, theta, XT, YT)
		# Calculate mean of 'H_grid' excluding the boundary zeros -> will feed into the convergence function
		mean_H_grid = np.mean(H_grid)
		if verbose:
			print(mean_H_grid)

		# Calculate dH/dt
		# Calculate 'dH_dt' based solely on the retinal drive
		dH_dt = H_grid_initial.copy()
		# Update dH/dt for each neuron to include the effects of the post-threshold activities of the surrounding neurons
		dH_dt = dH_dt + sp.signal.convolve2d(H_star, tectal_modifiers, mode='same', boundary='fill', fillvalue=0)
		# Update 'dH_dt' to reflect membrane decay
		dH_dt += alpha * H_grid

		# Update H_grid based on dH_dt
		H_grid = H_grid + dH_dt * dt
		# Create a copy of 'H_grid'
		new_H_grid = H_grid.copy()
		# Calculate  mean of 'new_H_grid' excluding boundary zeros
		mean_new_H_grid = np.mean(new_H_grid)
		# Has the tectal activity converged?
		converged = convergence(mean_new_H_grid, mean_H_grid)
		if verbose:
			print(mean_new_H_grid)
			n_loops =+ 1
			if n_loops > 20:
				break
			print(converged)
			time.sleep(1)

	return new_H_grid


# Update synaptic weightings
@cython.boundscheck(False)
cdef inline np.ndarray[double, ndim=2, mode='c'] update_synaptic_weightings(np.ndarray[double, ndim=2, mode='c'] new_H_grid, double theta, 
																			int XT, int YT, retinal_neurons, double h, 
																			np.ndarray[double, ndim=2, mode='c'] s, double epsilon):
	'''
	Update the synaptic weightings in 's'

	Description
		- After 'H_grid' (the tectal sheet depolarisation) has converged, the 
		relative depolarisation of each tectal neuron is used to influence the 
		strength of the synaptic connection between the activated retinal 
		neurons and the tectal nueron in question (Hebbian plasticity)
		- Thresholding is applied to 'new_H_grid' to generate 'new_H_star'
		- 'new_H_star' is then reshaped into a linear array ('lin_new_H_star') 
		after the boundary zeros are removed
		- For each tectal nueron in 'lin_new_H_star', if the post-threshold 
		potential is greater than the modification threshold ('epsilon') then 
		the synapse between the activated retinal neurons ('RN1' and 'RN2') is 
		increased in proportion to the tectal neuron depolarisation

	Parameters
		new_H_grid - numpy.ndarray of post-convergence tectal depolarisations
		theta - float threshold value
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		retinal_neurons - np.array containing 'RN1', 'RN2', integers identifying
						the columns of 's' that correspond to the selected
						retinal neurons
		h - float that sets the rate of modification
		s - numpy.ndarray containing the stregnth of the synaptic connections 
			between each retinal neuron and every tectal neuron
		epsilon - float modification threshold

	Returns: 's' - numpy.ndarray containing the stregnth of the synaptic 
	connections between each retinal neuron and every tectal neuron

	'''
	cdef np.ndarray[double, ndim=2, mode='c'] new_H_star, lin_new_H_star
	cdef int y, x

	# Apply thresholding to 'new_H_grid'
	new_H_star = threshold(new_H_grid, theta, XT, YT)
	# Convert 'new_H_star' into a linear array, removing boundary zeros
	lin_new_H_star = np.reshape(new_H_star, (XT*YT, 1))
	# Update 's'
	for y in range(XT*YT):
		if lin_new_H_star[y] > epsilon:
			for x in retinal_neurons:
				s[y, x] += h * lin_new_H_star[y]

	return s

# Cython Run Function
@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2, mode='c'] _run(double ND_mean, double ND_sd, int XT, int YT, int XR, int YR, str PM_type, bool default_polarity_markers, 
											double PM_strength_increase, str activity_pattern, double dt, double beta, double gamma, double delta, 
											double alpha, double theta, double h, int num_loops, bool verbose, double epsilon):
	'''
	Run Function

	Description
		- This function executes a script for retinotopic map formation between a 
		retinal sheet of neurons (dimensions: 'YR' by 'XR') and a sheet of 
		tectal neurons (dimensions: 'YT' by 'XT')

		- Functions prints 'X percent complete' for the deciles of completion
		
		- Function then plots the resulting 'fishnet' plot that indicates the
		centre of the receptive fields for the tectal neurons

	Parameters
		ND_mean - float mean of the normal distribution used to determine the 
				values in 's'
		ND_sd - float standard deviation of the aforementioned normal 
				distribution
		XT - integer x dimension of the tectal sheet (number of neurons)
		YT - integer y dimension of the tectal sheet (number of neurons)
		XR - integer x dimension of the retinal sheet (number of neurons)
		YR - integer y dimension of the retinal sheet (number of neurons)
		default_polarity_markers - boole, if 'True' PMs default to the top left 
									4 neurons on both the tectal and retinal 
									sheets. if 'False' then both retinal and 
									tectal polarity markers are selected at 
									random
		PM_strength_increase - float coefficient for the increase in stregnth of
							the synapses between retinal and tectal polarity
							markers
		activity_pattern - string determining which retinal activity activity_pattern function is
					used
		dt - float time step for the integration of the tectal interactions
		beta - float co-efficient of the excitatory interactions between neurons 
				that are 1 Manhattan distance away
		gamme - float co-efficient of the excitatory interactions between 
				neurons that are 2 Manhattan distance away
		delta - float co-efficient of the inhibitory interactions between 
				neurons that are 3 Manhattan distance away
		alpha - float tectal membrane decay constant
		theta - float threshold value
		h - float that sets the rate of modification
		num_loops - integer number of times retinal neurons are activated, the 
					tectal interactions are calculated, the tectal sheet 
					converges and the synaptic weightings in 's' are updated
		verbose - bool, prints outputs and time data when 'True'
		epsilon - float modification threshold


	Returns: 's' - numpy.array containing the stregnth of the synaptic 
	connections between each retinal neuron and every tectal neuron
	'''
	t1 = time.time()

	cdef np.ndarray[double, ndim=2, mode='c'] H_grid_initial, new_H_grid
	cdef int loop_count

	s = initialise_synaptic_connections(ND_mean, ND_sd, XT, YT, XR, YR)
	s = implement_polarity_markers(PM_type, s, default_polarity_markers, XR, YR, XT, YT, PM_strength_increase)
	s = normalise(s, ND_mean, XT, YT, XR, YR)
	loop_count = 0
	while loop_count < num_loops:
		retinal_neurons = activate_retinal_neurons(activity_pattern, XR, YR, loop_count)
		H_grid_initial = activate_tectal_sheet(XT, YT, retinal_neurons, s)
		new_H_grid = tectal_interactions(H_grid_initial, XT, YT, dt, beta, gamma, delta, alpha, theta, verbose)
		s = update_synaptic_weightings(new_H_grid, theta, XT, YT, retinal_neurons, h, s, epsilon)
		s = normalise(s, ND_mean, XT, YT, XR, YR)

		if loop_count%(num_loops//10) == 0:
			percent_complete = (float(loop_count)/float(num_loops)) * 100
			print ('%d percent complete' % percent_complete)

		loop_count += 1

	t2 = time.time()
	# Print run() time
	print ('Ran for %.2fs' % (t2-t1))
	return s

# Python Run Function
def run(double ND_mean, double ND_sd, int XT, int YT, int XR, int YR, str PM_type, bool default_polarity_markers, 
		double PM_strength_increase, str activity_pattern, double dt, double beta, double gamma, double delta, 
		double alpha, double theta, double h, int num_loops, bool verbose, double epsilon):
	return _run(ND_mean, ND_sd, XT, YT, XR, YR, PM_type, default_polarity_markers, 
				PM_strength_increase, activity_pattern, dt, beta, gamma, delta, alpha, theta, 
				h, num_loops, verbose, epsilon)


