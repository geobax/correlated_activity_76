# Computational Evaluation of the Neural Activity Model of Retinotopic Map Formation

A computational model was constructed in the Python™ and Cython programming languages and used to evaluate Willshaw and Malsburg’s neural activity theory of topographically organised network development. The investigation focuses on the development of retinotopic maps in the optic tectum of fish and amphibians. This study demonstrates that the neural activity model is able to successfully explain the systems-matching results observed in surgical mismatch experiments and that the model is robust to different patterns of retinal activity. Eight different types of retinal activity pattern are investigated, animations of map formation are produced and a novel function is developed to allow quantitative assessment of the retinotopic maps produced by the model.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

**1.** Download sim.py, sim.pyx, setup.py, build.sh and gif_animation.py

**2.** Open a terminal at the relevant folder and type:
```
bash build.sh
```	
This calls setup.py, which compiles the sim.pyx Cython file into C

**3.** Open sim.py and enter your chosen input parameters at the bottom of the script. Initial paramteres you may wish to vary include:

* The dimensions of the retinal and tectal sheets: XR, YR, XT & YT. Note these sheets must be square.

* 'PM_type'; the type of polarity marker, of which there are 4 style:
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
* 'r.npy' file containing synaptic weight matrices, periodically saved as the final map of the group develops

**6.** Open the terminal at the relevent folder and type:
```
python gif_animation.py
```
This produces and saves a 5 second gif of the development of the retinotopic map that 'r.pny' corresponds too

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

