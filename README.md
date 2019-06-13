# CudaSobel
Implementation of a sobel filter on CUDA c (using GPU).

There are two kernel in this project, the fist convert input image into gray scale (using 4 streams to emprove memory transfer), the second apply sobel filter.
