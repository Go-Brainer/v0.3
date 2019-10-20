# Requirements for Project Two: Go BOT model 2
## CSUF Fall 2019: CPSC 481 - Artificial Intelligence with Dr. William McCarthy

- [ ] Create the generate_mcts_games file, which generates games by encoding the game state before each move, encodes the move as a one-hot vector, and applies it.
- [x] Implement Monte-Carlo tree search, alpha-beta pruning and minimax (ch. 4). Create an MCTSAgent and let it play against itself.
- [ ] Create a program to create, and run Go games, and save them. Use it to generate 20 9x9 Go games, and store the features in features.py, and the labels in labels.py.
- [ ] Confirm the CNN from listings 6.24-26 of your text runs, and produces the output shown in your text. Print out the probabilities of its recommended moves (see 6.26).
- [ ] Groups with no liberties are removed from the board
- [ ] Create the KGSIndex class that downloads SGF files from https://u-go.net/gamerecords, and download the files. (see listing 7.1).
- [ ] Replay the (pretend) game from Listing 7.2. Make sure it replays the game correctly.
- [ ] Create the Go data processor that can transform raw SGF data into features and labels for a machine learning algorithm.
- [ ] Create the OnePlaneEncoder and SevenPlaneEncoder, and verify that they produce the correct output from the text.
- [ ] Create the training and test generators that use the GoDataProcessor, so that Keras can use those generators to fit the model and to evaluate it.
- [ ] Add the Adagrad optimizer to allow adaptive gradient methods.
- [ ] Train your Go bot using different hyperparameters to get best performance.
- [ ] Be written in Python. No issues are shown in PyCharm (all source code screens shown a green checkmark at the top right hand corner).
- [ ] Project directory pushed to a new GitHub repository listed above
