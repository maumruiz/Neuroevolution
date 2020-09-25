# Neuroevolution
This project contains the code for neuroevolution, a genetic algorithm approach to obtain the weights of a neural networks which solve an objective function (unknown to the user).

### Dependencies

* Python 3.6 +
* NumPy
* Pandas

### Usage

To run the genetic algorithm, cd into this repo's `src` root folder and execute:
    `python GANeuroevolution.py`.

There are some arguments that can be passed to the python execution in order to change the parameters given to the genetic algorithm. These arguments are listed on the section below. Here are some examples of how it can be used:

`python GANeuroevolution.py --generations 50`
`python GANeuroevolution.py --population 200`
`python GANeuroevolution.py --crossover_rate 0.95 --mutation_rate 0.15`
`python GANeuroevolution.py --generations 50 --mutation_rate 0.15`

### Arguments
The script takes the following arguments for the options

- `-g, --generations`: Number of generations, default is 150
- `-p, --population`: Population size, default is 100
- `-cr, --crossover_rate`: Crossover rate, default is 0.8
- `-mr, --mutation_rate`: Mutation rate, default is 0.3
