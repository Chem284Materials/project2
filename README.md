# Project 2 - Implementing Neural Network in CUDA

In this repository is a code a code that "discovers" the Morse potential via a feedforward neural network and backpropagation.
You'll now work towards getting the code running efficiently on GPUs.
As you'll note, this `README.md` doesn't include much specific information on exactly what you should do.
Use the skills and knowledge you've gained over the course of this semester to make intelligent decisions wherever the instructions leave room for interpretation.

## Task 1 - Port the Code to GPUs

The code currently runs on the CPU.
Rewrite it to run on GPUs, making an effort optimize for efficiency.

## Task 2 - Profile the Code

Perform profiling tests on Perlmutter, including analysis of Roofline plots.
Analyze the time cost of training your model with respect to the number of hidden layers, the size of the hidden layers, and the size of the training dataset.
Include your data and plots here, and explain your conclusions regarding the profiling results.

## Task 3 - Increase the Size of the Calculation

Note that the Morse potential problem is too small to effectively utilize Perlmutter's resources.
Modify your code to solve a more physically complex problem that can better utilize the Perlmutter GPUs.
For example, what if instead of trying to learn a Morse potential for a two-body system, you tried to learn a potential energy surface for a larger chemical system?
You may select a physical problem unrelated to molecular dynamics, if you prefer.
Provide this code **in addition** to the code for Tasks 1 and 2; in other words, submit code that solves the new problem as well as code that discovers the Morse potential.

Repeat your Perlmutter profiling calculations with the new system and discuss your results.

## Task 4 - Discuss the Code

Discuss your code's parallelization strategy.
Why did you choose this strategy?
In what ways could the code's performance be improved?
Describe some ways in which your neural network implementation would need to change to accomodate machine learning in the context of a condensed-phase molecular dynamics simulation involving thousands of atoms.

Your response to this task should be fairly extensive (>1,000 words).

## Collecting Roofline Data on Perlmutter

For the purpose of collecting data for the roofline plots, you can reference the following Perlmutter submission script.

```bash
#!/bin/bash
#SBATCH --job-name=neuralnet_roofline
#SBATCH --account=<insert_account_name>
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=neuralnet_roofline_%j.out
#SBATCH --error=neuralnet_roofline_%j.err

module load python
module load cudatoolkit

# Activate conda environment with PyCUDA
# You'll need to have previously created the environment with something like:
# conda create -n pycudaenv python=3.11 pip numpy pycuda
conda activate pycudaenv

srun --ntasks-per-node=1 dcgmi profile --pause
ncu \
    --set roofline \
    --section SpeedOfLight_RooflineChart \
    --launch-skip 0 \
    --launch-count 20 \
    --target-processes all \
    --force-overwrite \
    -o roofline_neuralnet \
    python neuralnet.py
srun --ntasks-per-node=1 dcgmi profile --resume
```

Note that this script uses `--launch-count` to restrict the number of kernel launches that are profiled.
You may need to make adjustments to this and other options in order to adapt the script to your code.

If the script runs successfully, it should produce a file called `roofline_neuralnet.ncu-rep`.
You can then generate the roofline plot by running, for example, `ncu-ui roofline_neuralnet.ncu-rep`.
Note that by default, running GUI applications on Perlmutter won't work; you can instead copy the file to your local machine before running `ncu-ui`.

## Answers

