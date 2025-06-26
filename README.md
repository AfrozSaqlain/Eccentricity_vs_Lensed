# A guide to Files
- `gen.py`: This is used to generate qtransform spectrograms of synthetic GW signals assuming aLIGO sensitivity. Sampling frequency is 4096 Hz.
- `data_2`: This is the data sample file.
- `tranformer.py` and `cnn.ipynb`: These are the neural network code where I have defined the model's structure and done training and testing.

There's one error after I updated the generation code: 
`ERROR: the A22 peak time is negative, dynamics is too short
ERROR(TEOBResumS): ringdown failed.`
