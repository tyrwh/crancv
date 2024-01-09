# crancv
Cranberry external image analysis scripts via python/OpenCV.

This image analysis pipeline was written by Breeding Insight, a USDA-funded program based at Cornell University, to support the USDA cranberry breeding programs.

## DEMO
To run an interactive demo illustrating the steps of the pipeline, click the `Binder` button above. This will launch a Jupyter notebook where you can walk through the steps on a sample image.

## SETUP
This code is designed to run from within a [Conda](https://anaconda.org/anaconda/conda) environment. We strongly recommend using [mamba](https://mamba.readthedocs.io/), a much faster implementation of the command-line `conda` tool. All `mamba` commands have the same structure and arguments as the relevant `conda` commands.

To install and set up `mamba`:
1. Download the appropriate installer and complete the installation process for your system
2. Open the terminal and confirm that mamba is installed with `mamba -h`
3. Find the correct YAML file for your OS
4. Create a new mamba environment from the appropriate YAML file:
   
`mamba env create -f yml/mamba_env_bicran.yml`

5. Try activating the environment:

`mamba activate bicran`

## RUNNING THE PIPELINE
Each time you run the pipeline, you will need to activate the `mamba` environment from your command-line environment:
`mamba activate pcv`

Once the environment is activated, you are ready to run the script. To list the potential commands:

`python CranExternalV1.py -h`

The only required argument is an input directory or image: If specifying a single image, it must of format .jpg, .jpeg, .tif, .tiff, or .png (case insensitive). If specifying a directory, the directory must contain at least one image of this format.


`python ./CranExternalV1.py -i test_dir/`

`python ./CranExternalV1.py -i test_image.png`

By default, the script will automatically name the output file based on the image or directory name. To change this, use the `-o` flag:

`python ./CranExternalV1.py -i test_dir -o results.csv`

In most cases, you should have a circular size marker visible in the image. To convert the output into cm, specify the diameter (in cm) with the `-s` flag:

`python ./CranExternalV1.py -i test_dir -o results_in_cm.csv -s 2.5`

Annotated images can be useful for validating that the pipeline is recognizing and segmenting objects correctly, and for troubleshooting if and when you encounter errors. To save out annotated images, either use the `-w` flag, which will automatically create a new directory based on the input name, or use the `-a` flag to name your own directory:

`python ./CranExternalV1.py -i test_dir -w   # will create a dir called test_dir_annotated`
`python ./CranExternalV1.py -i test_dir -a more_annotations`

By default, the script only spawns a single process. To spawn more processes, use the `-n` flag. The number of processes cannot exceed your machine's CPU count as given by `multiprocessing.cpu_count()`:

`python ./CranExternalV1.py -i test_dir -n 4`

## QUESTIONS/COMMENTS  
Please address all questions to Tyr Wiesner-Hanks ([tw372@cornell.edu](mailto:tw372@cornell.edu))
