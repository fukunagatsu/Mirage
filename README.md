# Mirage
Mirage is ancestral genome estimation software with high accuracy based on a phylogenetic mixture model and a RER model.

## Version
Version 1.0.0 (2020/10/09)

## Usage
Mirage has two modes, a training mode and an estimation mode. In the training mode, Mirage trains the evolutionary model parameters and estimates ancestral genome from an input ortholog table and a phylogenetic tree using Mirage "train" command. Mirage "train" command requires 2 options ([-i InputFileName] and [-o OutputFileName]). In the estimation mode, Mirage only estimates ancestral genome from an input ortholog table, a phylogenetic tree and evolutionary model parameters using Mirage "estimate" command. Mirage "estimate" command requires 3 options ([-i InputFileName], [-o OutputFileName] and [-p ParameterFileName]).

## Example
    ./Mirage train -i archaea_data.txt -o output1
    ./Mirage estimate -i archaea_data.txt -o output2 -p output1.par

## Command and Options
    train: train evolutionary model parameters and estimate ancestral genome

    Mirage train [-i InputFileName] [-o OutputFileName] [-l MaximumGeneCopyNumbers]  
                [-m ModelId] [-k NumberOfMixtures] 
   
    Options:
    (Required)
        -i STR    InputFileName
        -o STR    OutputFileName
        
    (Optional) 
        -l INT    The number of maximum gene copy numbers [default:3]
        -m INT    Specification of the evolutionary model. 0: the RER model 1: the two-parameter model 2: the K&M model 3: the BDI model [default: 0]
        -k INT    The number of mixture components in the phylogenetic mixture model [default: 5]
        
    estimate: only estimate ancestral genome
    
    Mirage train [-i InputFileName] [-o OutputFileName] [-p ParameterFileName]
                
## External libraries
This repository includes the code of an external libraries, "Eigen".  
[Eigen](http://eigen.tuxfamily.org/index.php) is a C++ library for linear algebra.

## License
This software is released under the MIT License, see LICENSE.txt.  
Eigen is primarily licensed under MPL2, and please see [the original license description](Eigen/COPYING.README).

## Reference
Tsukasa Fukunaga and Wataru Iwasaki. "Mirage; A phylogenetic mixture model to reconstruct gene content evolutionary history using a realistic parameter model of gene gain and loss events." under submission.