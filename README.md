# Document Scanner

This repository contains the code for an exercise at our university in the lecture 'computer vision'.

## Goal

Implement a python script, which takes images from the `images/` directory as an input and executes a warp of perspective, imitating a classic document scanner.

## Usage

1. Add images of a document to the `images/` directory.
2. If needed, tweak some settings on the very top of the `main.py` file. By default, the detected corners, the search for the corners of the paper and the results are displayed. You can change this behavior to safe the results to the `images/` directory.
3. Make sure `opencv2` and `numpy` are installed.
4. Run the script with `python3 main.py`.

## Caveats

- The scanner uses a corner detection algorithm. Therefore the document must be positioned on a dark and homogeneous surface to avoid detecting false edges
