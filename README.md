# Go-Brainer
The Go-Brainer project is an assignment for CSUF Fall 2019 CPSC 481: Artificial Intelligence with Dr. William McCarthy

## Project Description
Use Deep Learning techniques from lectures and the book *Deep Learning and the Game of Go* by Max Pumperla and Kevin Ferguson to develop a artificial intelligence bot that can play the board game Go.

## Participants:
Nick Bernstein  skolyr@csu.fullerton.edu  
Alex Vidal      avidal@csu.fullerton.edu

## Usage
C:\PATH\>python Go-Brainer.py -h  
usage: Go-Brainer [-h] [-s SIZE] [-k KOMI] {r,d,h} {r,d,h}

An AI Go Bot

positional arguments:
  {r,d,h}               Choose Player Agent for black: r - random, d - depth
                        pruned minimax, h - human
  {r,d,h}               Choose Player Agent for black: r - random, d - depth
                        pruned minimax, h - human

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Board size is n by n. Must be between 5 and 19.
  -k KOMI, --komi KOMI  Sets komi. Must be between 0 and 10.
