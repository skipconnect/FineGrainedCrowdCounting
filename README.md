# FineGrainedCrowdCounting
Investigates the FineGrainedCrowdCounting task, evaluating and comparing different DL-architechtures performance in this task. This repository contains the following 2 folders:

* **CrowdCounting-P2P** which contains the implementation af P2P-network modified to the FGCC-task. Much of the code is a modication of https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
* **FineGrainedCountingCV** which implements the model of https://github.com/jia-wan/Fine-Grained-Counting and an implementation of FG-MC-OC (https://ieeexplore.ieee.org/document/9506384)



### Prerequisites

- Git: [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Anaconda or Miniconda: [Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Installation

1. Clone the repository:
   ```shell
   git clone https://github.com/skipconnect/FineGrainedCrowdCounting.git
   ```

2. Navigate to the project directory:
   ```shell
   cd FineGrainedCrowdCounting
   ```

3. Create a new conda environment using the `environment.yml` file:
   ```shell
   conda env create -f environment.yml
   ```

4. Activate the newly created environment:
   ```shell
   conda activate environment-name
   ```
   Replace `environment-name` with the name of the environment created in the previous step.

## Usage
- For implementations of (https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) go to 

   ```shell
   cd ./CrowdCounting-P2PNet/
   ```
  And follow instructions in README.md
  
- For implementations of https://github.com/jia-wan/Fine-Grained-Counting
   ```shell
   cd ./FineGrainedCountingCV/
   ```
  And follow instructions in README.md

- For implementations of https://ieeexplore.ieee.org/document/9506384
   ```shell
   cd ./FineGrainedCountingCV/FG-MC-OC/
   ```
  And follow instructions in README.md


   
