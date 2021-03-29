import numpy as np
cimport numpy as np
from game_functions import initialize_game, random_move, move_down, move_left, move_right, move_up, check_for_win, add_new_tile



cpdef ai_move(board, num_searches, search_length):
    moves = [move_up, move_down, move_left, move_right]
    cdef int[4] scores = [0, 0, 0, 0]
    cdef int i
    #cdef int[4][4] first_board
    cdef np.ndarray[np.int_t, ndim=2] first_board
    cdef bint move
    cdef int score
    cdef int move_number
    cdef int j
    cdef bint legal
    #cdef int[4][4] search_board
    cdef np.ndarray[np.int_t, ndim=2] search_board
    for i in range(4):
        
        first_board, move, score = moves[i](board)
        
        if move:
            first_board = add_new_tile(first_board)
            scores[i] += score
        else:
            continue
        for j in range(num_searches):
            move_number = 1
            search_board = np.copy(first_board)
            legal = True
            
            while legal and move_number < search_length:
                search_board, legal, score = random_move(search_board)
                if legal:
                    search_board = add_new_tile(search_board)
                    scores[i] += score
                    move_number += 1
                    
    search_board, legal, score = moves[np.argmax(scores)](board)
    return search_board, legal