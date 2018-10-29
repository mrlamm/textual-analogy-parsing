# Textual Analogy Parsing


Textual Analogy Parsing (TAP) is the task of identifying analogy frames from text. Analogy frames are a discourse-aware shallow semantic representation that highlights points of similarity and difference between analogous facts. 

One motivation for TAP is that it can be used to automatically plot quantitative facts. Given the sentence 

> According to the U.S. Census, almost 10.9 million African Americans, or 28%, live at or below the poverty line, compared with 15% of Latinos and approximately 10% of White Americans
 
a TAP parser outputs the following analogy frame:

<p align="center"> <img src="figures/avm.png" width=30></p>

This can be visualized straighforwardly, by assigning elements of the *compared content* (in the curly brackets) to the x- and y- axes of a plot, and assigning elements of the *shared content* (in the outer-tier of the frame) to plot elements like titles and axis labels:

<p align="center"> <img src="figures/plot.png" width=30></p>

## Dataset

We report experiments in the paper on a hand-annotated dataset of quantitative analogy frames identified in the Penn Treebank WSJ Corpus. 

The data are available for download here.

Some statistics: 

<p align="center"> <img src="figures/dataset_stats.png"></p>

Here, *Count* refers to the number of frames and *Length* refers to the number of values compared within a given frame. *Av(erage)* is the per-sentence average over a given dataset and *max(imum)* is the maximum over all sentences. *Tot(al)* is the total number of frames in a given dataset.

