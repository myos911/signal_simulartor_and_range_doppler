# GPU-Accelerated Range-Doppler Algorithm

This repository contains a GPU-accelerated implementation of the Range-Doppler algorithm used in Synthetic Aperture Radar (SAR). This has been achieved through using Simone Mencarelli's implementation of a point scatterer radar signal simulator and unoptimised version of the Range-Doppler image formation algorithm. 

## Instructions on Running the Project


### Requirements
This project has been tested and run with an NVIDIA Jetson Xavier NX 16GB Developer Kit — as part of the [reComputer Jetson-20](https://www.seeedstudio.com/Jetson-20-1-H2-p-5329.html).

This project was developed using **Python 3.8.0**, with the requirements used for this project listed in `requirements.txt`*.

_*Note that `requirements.txt` was built using the_ `pip3.8 freeze` _command_. 

### Running the Algorithm

First, clone the repository, or open it from the project compendium folder. Then:

1. Ensure that the folder `Simulation_Data` contains the file `data_dump.pk`. Please refer to `Simulation_Data/README.md` for instructions on how get this file.
2. Ensure that the folder `Antenna_Pattern` contains the file `gain_pattern.pk`. Please refer to `Antenna_Pattern/README.md` for instructions on how get this file.

3. From the terminal, you can run 

  ```
  python3.8 scriptedsympyr.py
  ```
  This script also has the following options:

  ```
  usage: scriptedsympyr [-h] [--dump-data] [--plot]

  Run the SAR simulator with dumping and plotting control.

  optional arguments:
  -h, --help   show this help message and exit
  --dump-data  Enable data dumping
  --plot       Enable plotting
  ```

  This will run the algorithm, and will output the benchmarked results. 


### Running the Tests

To run the tests, do the following:

1. Ensure that the `original_data` folder is populated. To populate this data, please refer to `original_data/README.md` for instructions.

2. Then you can run:
   ```
   python3.8 test_output.py
   ```

   This will run the test cases associated with each step in the algorithm. 

-- Oct. 2023, Matan Yosef & Jia Tee--
