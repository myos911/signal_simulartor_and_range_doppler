# Original Data - The Original Output of Each Core Step in the Algorithm

This folder is where the original data created by the unoptimised version of the Range-Doppler algorithm are dumped. It is used for the assertions in `test_output.py`. 
These dumps are too large to be tracked by git, and are hence not in the initial repository. 

## Populating via Copy-Paste
GitHub does not allow files that are over a certain size. In order to populate this file, you must paste in the correct files from the project compendium folder called `original_data`. Please copy and paste the following files into this folder:
* `matched_filter.pk` - Encapsulates Step 1 of Range Doppler (Range Comrpession)
* `azimuth_fft.pk` - Encapsulates Step 2 of Range Doppler (Azimuth FFT)
* `rcmc.pk` - Encapsulates Step 3 of Range Doppler (RCMC)
* `azimuth_filter.pk`- Encapsulates Step 4 of Range Doppler (Azimuth Compression)
* `azimuth_ifft.pk`- Encapsulates Step 5 of Range Doppler (Azimuth IFFT)


## Using the Data
This data is used in the assertions for `test_output.py`. It is used to compare against the dumps inside the `original_data` folder. To run this test, you can enter the following in the terminal (whilst in the repository root folder):
```
python3.8 test-output.py
```
