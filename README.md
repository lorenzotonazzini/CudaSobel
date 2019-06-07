# CudaSobel
Implementation of a sobel filter on CUDA c (using GPU).

There are two kernel in this project, the fist convert input image into gray scale, the second apply sobel filter
using pinned memory (non paginable) and constant to emprove memory transfer.
