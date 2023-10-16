# Optimised Data - The Output of Each Core Step in the Algorithm

This folder is where data dumps are saved. It is used for the assertions in `test_output.py`. 
These dumps are too large to be tracked by git, and are hence not in the initial repository. 

## Option A: Populating via running the algorithm
In order to populate this file, please enter the following in terminal (whilst in the repository root folder):
```
python3.8 scriptedsympyr.py --dump-data
```

This will run the exisiting Range-Doppler algorithm inside this repository, and populate this folder. 

## Option B: Populating via running tests
Alternatively, you can run:
```
python3.8 test_output.py
```
**However,** this requires that the folder `original_data` is also populated. Please refer to `original_data/README.md` for instructions on how to populate this. 

<hr>

To make sure that this has been populated corrected, the following files should appear:
* `matched_filter.pk` - Encapsulates Step 1 of Range Doppler (Range Comrpession)
* `azimuth_fft.pk` - Encapsulates Step 2 of Range Doppler (Azimuth FFT)
* `rcmc.pk` - Encapsulates Step 3 of Range Doppler (RCMC)
* `azimuth_filter.pk`- Encapsulates Step 4 of Range Doppler (Azimuth Compression)
* `azimuth_ifft.pk`- Encapsulates Step 5 of Range Doppler (Azimuth IFFT)
