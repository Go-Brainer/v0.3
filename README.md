# Go-Brainer
The Go-Brainer project is an assignment for CSUF Fall 2019 CPSC 481: Artificial Intelligence with Dr. William McCarthy

## Project Description
Use Deep Learning techniques from lectures and the book *Deep Learning and the Game of Go* by Max Pumperla and Kevin Ferguson to develop a artificial intelligence bot that can play the board game Go.

## Participants:
Nick Bernstein  skolyr@csu.fullerton.edu  
Alex Vidal      avidal@csu.fullerton.edu

## Usage
C:\PATH\>python Go-Brainer.py -h
usage: Go-Brainer [-h] [-s SIZE] [-m {b,w}] [-k KOMI]

An AI Go Bot

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Board size is n by n. Must be between 5 and 19.
  -m {b,w}, --human {b,w}
                        Enables human v. bot mode and sets human's color.
  -k KOMI, --komi KOMI  Sets komi. Must be between 0 and 10.