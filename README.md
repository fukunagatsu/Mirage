
# Mirage
Mirage is ancestral genome estimation software with high accuracy based on a phylogenetic mixture model.

## Version
Version 2.0.0 (2022/04/04)

## Usage
Mirage has two modes, a training mode and an estimation mode. In the training mode, Mirage trains the evolutionary model parameters and estimates ancestral genome from an input ortholog table and a phylogenetic tree using Mirage "train" command. Mirage "train" command requires 2 options ([-i InputFileName] and [-o OutputFileName]). In the estimation mode, Mirage only estimates ancestral genome from an input ortholog table, a phylogenetic tree and evolutionary model parameters using Mirage "estimate" command. Mirage "estimate" command requires 3 options ([-i InputFileName], [-o OutputFileName] and [-p ParameterFileName]).

## Example
    ./Mirage train -i archaea_data.txt -o output1
    ./Mirage estimate -i archaea_data.txt -o output2 -p output1.par

## Command and Options
    train: train evolutionary model parameters and estimate ancestral genome

    Mirage train [-i InputFileName] [-o OutputFileName] [-l MaximumGeneCopyNumbers]  
                [-m Gain/LossModelId] [-k NumberOfMixtures] [-n HeterogeneityModelId]  
                [-s RandomSeed] [-r OutputStyle]  
   
    Options:
    (Required)
        -i STR    InputFileName
        -o STR    OutputFileName
        
    (Optional) 
        -l INT    The maximum gene copy numbers [default:3]
        -m INT    Specification of the gain/loss model. 0: the BDARD model 1: the BD model 2: the C&M model 3: the BDI model [default: 0]
        -k INT    The number of mixture components in the phylogenetic mixture model [default: 5]
        -n INT    Specification of the heterogeneity model. 0: the PPM model 1: the PDF model 2: the Gamma model 3: the DPM model [default: 3]
        -s INT    A seed value for randomization. When the value is set to negative, the seed value is randomly generated. [defualt: -1]
        -r INT    The output style of gene content reconstruction. When the value is 0, the numbers of genes in the ancestral nodes are output. When the value is 1, the numbers of increase or decrease of genes in the branches are output. [defualt: 0]

    estimate: only estimate ancestral genome
    
    Mirage train [-i InputFileName] [-o OutputFileName] [-p ParameterFileName] [-r OutputStyle]

    Options:
    (Required)
        -i STR    InputFileName
        -o STR    OutputFileName
        -p STR    ParameterFileName
        
    (Optional) 
        -r INT    The output style of gene content reconstruction. [defualt: 0]
                
## Input File Format
The input file must be in the following format.  
In the first, second and third lines, the number of ortholog, the phylogenetic tree in the newick format and the taxon name in the phylogenetic tree is described, respectively. In the following lines, the ortsholog table is described.
We have uploaded three files, archaea_data.txt, micrococcales_data.txt, and fungi_data.txt, as the input file examples, please see them.

## Output File Format
In the training mode, Mirage output four files whose extensions are "bas", "par", "res" and "ahr". In the estimation mode, Mirage output three files, "bas", "res" and "ahr" file. If users specify "output1" in command line option "-o", the output file is "output1.bas", "output1.par", "output1.res" and "output1.ahr". 

"bas" file describes basic information about the execution, e.g. runtimes option and likelihood. "res" files describes responsibilities, which are the probabilities that each ortholog belongs to each gene-content cluster. "ahr" files describes estimated gene content history. The gene contets are outputted in preorder traversal (current-left-right) of the input phylogenetic tree.

"par" file describes estimated evolutionary model parameters. The first line describes parameter $\phi$, mixing probabilites of gene-content clusters. From the next line, pi_1, R_1, pi_2, R_2, ..., pi_k, R_k are described. In the R_i, the parameters are listed in the order [R_i]01, [R_i]12, ..., [R_i] (l-1)l, [R_i]10, [R_i]21, ..., [R_i]l(l-1). Here, l is the maximum gene copy numbers.

## External libraries
This repository includes the code of an external libraries, "Eigen".  
[Eigen](http://eigen.tuxfamily.org/index.php) is a C++ library for linear algebra.

## License
This software is released under the MIT License, see LICENSE.txt.  
Eigen is primarily licensed under MPL2, and please see [the original license description](Eigen/COPYING.README).

## Reference
Tsukasa Fukunaga and Wataru Iwasaki. "Mirage; A phylogenetic mixture model to reconstruct gene content evolutionary history using a realistic parameter model of gene gain and loss events." Bioinformatics Advances, vbab014, (2021)

Tsukasa Fukunaga and Wataru Iwasaki. "Mirage 2.0: fast and memory-efficient reconstruction of gene-content evolution considering heterogeneous evolutionary patterns among ortholog groups." under submission.
