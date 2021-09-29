# certainty-estimator

## Intro
certainty-estimator is a package used to estimate the certainty of scientfic findings. It is released with
EMNLP 2021 paper `Measuring Sentence-level and Aspect-level (Un)Certainty in Science Communications` by [Jiaxin Pei](https://jiaxin-pei.github.io/) and [David Jurgens](https://jurgens.people.si.umich.edu/).


## Install 

### Use pip
If `pip` is installed, certainty-estimator could be installed directly from it:

    pip3 install certainty-estimator

### Dependencies
	python>=3.6.0
	torch>=1.6.0
	transformers >= 3.1.0
	numpy
	math
	tqdm
	
	

    

## Estimating sentence-level certainty

### Notes: During your first usage, the package will download a model file automatically, which is about 500MB.

### `Construct the Predictor Object`
	>>> from certainty_estimator.predict_certainty import CertaintyEstimator
	>>> estimator = CertaintyEstimator('sentence-level')
Cuda is disabled by default, to allow GPU calculation, please use

	>>> from certainty_estimator.predict_certainty import CertaintyEstimator
	>>> estimator = CertaintyEstimator('sentence-level',cuda=True)

### `predict`
`predict` is the core method of this package, 
which takes a single text or a list of texts, and returns a list of raw values in `[1,6]` (higher means more certain, while lower means less).

	# Predict certainty for a single scientific finding
	>>> text = 'The reason for this might be that the fetal central nervous system, which controls movements in general and facial movements in particular did not develop at the same rate and in the same manner as in fetuses of mothers who did not smoke during pregnancy.'
	>>> estimator.predict(text)
	[2.6891987]
	
	# Predict certainty for a list of scientific finding
        >>> text = ['The reason for this might be that the fetal central nervous system, which controls movements in general and facial movements in particular did not develop at the same rate and in the same manner as in fetuses of mothers who did not smoke during pregnancy.', 'Mice lacking the tet1 gene were able to learn to navigate a water maze, but were unable to extinguish the memory.']
	>>> estimator.predict(text)
	[2.6891987, 5.01066]
	
	# when calculating certainty for a long list of findings, use the following code to display the progress
  	>>> from tqdm import tqdm
	>>> text = [a long list of findings]
	>>> estimator.predict(text,tqdm=tqdm)
  	[2.6891987, 5.01066, ... ,4.28066, 5.77066]
  
  
  
## Estimating aspect-level certainty

### Notes: During your first usage, the package will download a model file automatically, which is about 500MB.

### `Construct the Predictor Object`
	>>> from certainty_estimator.predict_certainty import CertaintyEstimator
	>>> estimator = CertaintyEstimator('aspect-level')
Cuda is disabled by default, to allow GPU calculation, please use

	>>> from certainty_estimator.predict_certainty import CertaintyEstimator
	>>> estimator = CertaintyEstimator('aspect-level',cuda=True)

### `predict`
`predict` is the core method of this package, 
which takes a single text or a list of texts, and returns a list of raw values in `[1,6]` (higher means more certain, while lower means less).

	```# Predict certainty for a single scientific finding
	>>> text = 'Mice lacking tet1 had much lower levels of hydroxymethylation -- an intermediate step in the removal of methylation -- in the hippocampus and the cortex, which are both key to learning and memory.'
	>>> result = estimator.predict(text)
        >>> result 
	[[('Extent', 'Uncertain'), ('Probability', 'Certain')]]
	
	# Predict certainty for a list of scientific finding
        >>> text = ['Mice lacking tet1 had much lower levels of hydroxymethylation -- an intermediate step in the removal of methylation -- in the hippocampus and the cortex, which are both key to learning and memory.', 'Dopamine and serotonin are important for different forms of flexibility associated with receiving reward and punishment.']
	>>> result = estimator.predict(text, get_processed_output = True)
        >>> result 
	[[('Extent', 'Uncertain'), ('Probability', 'Certain')], [('Probability', 'Certain')]]
  
 
	# when calculating certainty for a long list of findings, use the tqdm to display the progress
        >>> from tqdm import tqdm
	>>> text = [a long list of findings]
	>>> estimator.predict(text,tqdm=tqdm)```



## Contact
Jiaxin Pei (pedropei@umich.edu)
