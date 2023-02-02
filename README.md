# STitch3D

### Construction of a 3D whole organism spatial atlas by joint modeling of multiple slices

![STitch3D\_pipeline](Overview.jpg)

<p align = "justify"> 
We developed STitch3D, a deep learning-based method for 3D reconstruction of tissues or whole organisms. Briefly, STitch3D characterizes complex tissue architectures by borrowing information across multiple 2D tissue slices and integrates them with a paired single-cell RNA-sequencing atlas. With innovations in model designs, STitch3D enables two critical 3D analyses: First, STitch3D detects 3D spatial tissue regions which are related to biological functions, for example cortex layer structures in brain; Second, STitch3D infers 3D spatial distributions of fine-grained cell types in tissues, substantially improving the spatial resolution of seq-based ST approaches. The output of STitch3D can be further used for various downstream tasks like inference of spatial trajectories, denoising of spatial gene expression patterns, identification of genes enriched in specific biologically meaningful regions and detection of cell type gradients in newly generated virtual slices.
</p>

An example: STitch3D reconstructed the mouse brain, successfully detected 3D layer organizations of the cerebral cortex, and accurately reconstructed curve-shaped distributions of four hippocampal neuron types in three cornu ammonis areas and dentate gyrus.
![hpc](mouse_brain_layers.gif) ![hpc](mouse_brain_hpc.gif)

Installation
------------

Tutorials and reproducibility
-----------------------------

Interactive 3D results
----------------------
