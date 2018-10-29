# Textual Analogy Parsing


Textual Analogy Parsing (TAP) is the task of identifying analogy frames from text. Analogy frames are a discourse-aware shallow semantic representation that highlights points of similarity and difference between analogous facts. 

One motivation for TAP is that it can be used to automatically plot quantitative facts. Given the sentence 

> According to the <span style="color:yellow">U.S.Census</span>, almost 10.9 million African Americans, or 28%, live at or below the poverty line, compared with 15% of Latinos and approximately 10% of White Americans
 
a TAP parser outputs the following analogy frame:



This can be visualized straighforwardly, by assigning elements of the *compared content* (in the curly brackets) to the x- and y- axes of a plot, and assigning elements of the *shared content* (in the outer-tier of the frame) to plot elements like titles and axis labels:

     
        
