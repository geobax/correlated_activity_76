# Computational Evaluation of the Neural Activity Model of Retinotopic Map Formation

A computational model was constructed in the Python and Cython programming languages and used to evaluate Willshaw and Malsburg’s neural activity theory of topographically organised network development. The investigation focuses on the development of retinotopic maps in the optic tectum of fish and amphibians. This study demonstrates that the neural activity model is able to successfully explain the systems-matching results observed in surgical mismatch experiments and that the model is robust to different patterns of retinal activity. Eight different types of retinal activity pattern are investigated, animations of map formation are produced and a novel function is developed to allow quantitative assessment of the retinotopic maps produced by the model.

I recommend reading *["How Patterned Neural Connections Can Be Set Up By Self-Organisation"](http://rspb.royalsocietypublishing.org/content/royprsb/194/1117/431.full.pdf)* by *D.J. Willshaw & C. von der Malsburg, 1976*, before exploring this model.

The code is designed to be read in conjunction with the **'project_report.pdf'** file in this repository.

## Getting Started

These instructions will allow you to get a copy of the project up and running on your local machine for development and testing purposes.

**1.** Download:
* sim.py
* sim.pyx
* setup.py
* build.sh
* gif_animation.py

**2.** Open a terminal at the relevant folder and type:
```
bash build.sh
```	
This calls setup.py, which compiles the sim.pyx Cython file into C.

**3.** Open sim.py and enter your chosen input parameters at the bottom of the script. Initial paramteres you may wish to vary include:

* The dimensions of the retinal and tectal sheets: 'XR', 'YR', 'XT' & 'YT'. Note these sheets must be square.

* 'PM_type'; the type of polarity marker, of which there are 4 styles:
	* 'none'. No polarity markers are implemented
	* 'square'; this can be further modified by varying 'default_polarity_markers' between 'True' and 'False'
		* 'True' defaults the polarity markers to the centre of the retinal and tectal sheets. This is polarity marker style 1
		* 'False' means the positioning of the polarity markers on the retinal and tectal sheets is chosen at random. This is polarity marker style 2
	* 'graded': synaptic strength is increased based on the relative positions of the retinal and tectal neurons in their respective neural sheets. This is polarity 	marker style 3 

* 'activity_pattern'; the pattern of retinal activity that is simulated. There are 8 different types built into the model:
	* 'pairs'
	* '2_pairs'
	* 'singles'
	* '2_singles'
	* 'squares'
	* 'sweep'
	* 'strobe'
	* 'ocular_dominance'

	Note the neural firing threshold ('theta') and neural modification threshold ('epsilon') need to be increased in accordance with the number of retinal neurons active as part of the activity pattern (this is detailed in each of the activity pattern functions)

**4.** Open a terminal at the relevent folder and type:
```
python sim.py
```

**5.** Once the script has executed the following are saved:
* The plots of the retinotopic maps produced
* 'map_qualities.txt' file of the qualities of each map, as well as the group's mean and stadard deviation in quality
* 'r.npy' file containing synaptic weight matrices, periodically saved, as the final map of the group develops

**6.** Open the terminal at the relevent folder and type:
```
python gif_animation.py
```
This produces and saves a 5 second GIF of the development of the retinotopic map that 'r.pny' corresponds to.

### Example Animation of Retinotopic Map Formation

This is an animation of the formation of a retinotopic map between a 10x10 neuron retinal sheet and a 10x10 tectal sheet, modeled using polarity marker style 1, 'pairs' activity pattern and 500,000 iterations.

![alt text](https://github.com/geobax/correlated_activity_76/blob/master/animation.gif)

## Prerequisites

The easiest way to get up and running is to install Python 2.7 via [Anaconda](https://www.continuum.io/downloads), as this comes packaged with the Numpy and Scipy libraries that are used extensively in this model.

Other prerequisites include:

* [Cython](http://cython.org) v.0.24
* [GCC](https://gcc.gnu.org) v.7.1
* [ImageMagick](https://www.imagemagick.org) 


## Author

**George Christopher Baxter**

## Acknowledgments

* The inspiration for this project is D.J. Willshaw & C. von der Malsburg's original 1976 paper _["How Patterned Neural Connections Can Be Set Up By Self-Organisation"](http://rspb.royalsocietypublishing.org/content/royprsb/194/1117/431.full.pdf)_.
* I would like to thank my supervisor, Dr. Stephen Eglen, for his support, guidance and patience; without his help this project would not have come to fruition. 
* I would also like to thank Professor David Willshaw, whose personal correspondence helped me to spot an error in the code that was preventing retinotopic map formation.

