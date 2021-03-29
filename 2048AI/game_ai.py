import numpy as np

from game_functions import initialize_game, random_move, move_down, move_left, move_right, move_up, check_for_win, add_new_tile

def ai_move(board):

    # Define search parameters
    num_searches = 200
    search_length = 20
    
    # Define move list, instantiate beginning score for each move as 0
    moves = [move_up, move_down, move_left, move_right]
    scores = [0, 0, 0, 0]
    
    # We must test a set of sample runs and establish a score for each move
    for i in range(4):
        # We use the game's functional "move_up", "move_down", etc. functions that normally are applied to a human input
        # first_board is an initial board state we create that is separate from the actual board to be played on. The
        # AI will use this board to experiment on
        # Score is the returned score, aka the running sum of the values of all tiles that have been combined
        first_board, move, score = moves[i](board)
        
        # If a move has been successfully made
        if move:
            first_board = add_new_tile(first_board)
            scores[i] += score
        else:
            continue
            
        # Now we run the Monte Carlo simulations. num_searches is the total number of searches the program will perform
        # on each possible move.
        for j in range(num_searches):
            move_number = 1
            search_board = np.copy(first_board)
            legal = True
            
            # search_length is the depth of each search performed. How many random moves does the AI perform per simulation?
            # The loop needs to break if the board has no legal moves remaining.
            while legal and move_number < search_length:
                # Make a random move 
                search_board, legal, score = random_move(search_board)
                
                # Evaluate it. If the position is legal, we add the move's score to our total score for that move function.
                if legal:
                    search_board = add_new_tile(search_board)
                    scores[i] += score
                    move_number += 1
                    
    # Now suggest the move that has accumulated the most total score across all testing. This is equal to the highest average
    # score since all moves are tested equally.
    search_board, legal, score = moves[np.argmax(scores)](board)
    return search_board, legal