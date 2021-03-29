import numpy as np
cimport numpy as np
cdef int POSSIBLE_MOVES_COUNT = 4
cdef int CELL_COUNT = 4
cdef int NUMBER_OF_SQUARES = CELL_COUNT * CELL_COUNT
cdef int[10] NEW_TILE_DISTRIBUTION = np.array([2, 2, 2, 2, 2, 2, 2, 2 ,2, 4])

cpdef initialize_game():
    cdef np.ndarray[np.int_t, ndim=1] board = np.zeros((NUMBER_OF_SQUARES), dtype="int")
    cdef np.ndarray[long long, ndim=1] initial_twos = np.random.default_rng().choice(NUMBER_OF_SQUARES, 2, replace=False)
    board[initial_twos] = 2
    cdef np.ndarray[np.int_t, ndim=2] new = board.reshape((CELL_COUNT, CELL_COUNT))
    return new

cpdef push_board_right(board):
    new = np.zeros((CELL_COUNT, CELL_COUNT), dtype="int")
    cdef bint done = False
    cdef int row
    cdef int count
    cdef int col
    for row in range(CELL_COUNT):
        count = CELL_COUNT - 1
        for col in range(CELL_COUNT - 1, -1, -1):
            if board[row][col] != 0:
                new[row][count] = board[row][col]
                if col != count:
                    done = True
                count -= 1
    return (new, done)


cpdef merge_elements(board):
    cdef int score = 0
    cdef bint done = False
    cdef int row
    cdef int col
    for row in range(CELL_COUNT):
        for col in range(CELL_COUNT - 1, 0, -1):
            if board[row][col] == board[row][col-1] and board[row][col] != 0:
                board[row][col] *= 2
                score += board[row][col]
                board[row][col-1] = 0
                done = True
    return (board, done, score)


cpdef move_up(board):
    cdef np.ndarray[np.int_t, ndim=2] rotated_board = np.rot90(board, -1)
    cdef np.ndarray[np.int_t, ndim=2] pushed_board
    cdef bint has_pushed
    pushed_board, has_pushed = push_board_right(rotated_board)
    cdef np.ndarray[np.int_t, ndim=2] merged_board
    cdef bint has_merged
    cdef int score
    merged_board, has_merged, score = merge_elements(pushed_board)
    cdef np.ndarray[np.int_t, ndim=2] second_pushed_board
    second_pushed_board, _ = push_board_right(merged_board)
    cdef np.ndarray[np.int_t, ndim=2] rotated_back_board = np.rot90(second_pushed_board)
    cdef bint move_made = has_pushed or has_merged
    return rotated_back_board, move_made, score

    
cpdef move_down(board):
    board = np.rot90(board)
    cdef bint has_pushed
    board, has_pushed = push_board_right(board)
    cdef bint has_merged
    cdef int score
    board, has_merged, score = merge_elements(board)
    board, _ = push_board_right(board)
    board = np.rot90(board, -1)
    cdef bint move_made = has_pushed or has_merged
    return board, move_made, score


cpdef move_left(board):
    board = np.rot90(board, 2)
    cdef bint has_pushed
    board, has_pushed = push_board_right(board)
    cdef bint has_merged
    cdef int score
    board, has_merged, score = merge_elements(board)
    board, _ = push_board_right(board)
    board = np.rot90(board, -2)
    cdef bint move_made = has_pushed or has_merged
    return board, move_made, score


cpdef move_right(board):
    cdef bint has_pushed
    board, has_pushed = push_board_right(board)
    cdef bint has_merged
    cdef int score
    board, has_merged, score = merge_elements(board)
    board, _ = push_board_right(board)
    cdef bint move_made = has_pushed or has_merged
    return board, move_made, score


cpdef fixed_move(board):
    move_order = [move_left, move_up, move_down, move_right]
    cdef np.ndarray[np.int_t, ndim=2] new_board
    cdef bint move_made
    for func in move_order:
        new_board, move_made, _ = func(board)
        if move_made:
            return new_board, True
    return board, False


cpdef random_move(board):
    cdef bint move_made = False
    cdef int move_index
    cdef int score
    move_order = [move_right, move_up, move_down, move_left]
    while not move_made and len(move_order) > 0:
        move_index = np.random.randint(0, len(move_order))
        move = move_order[move_index]
        board, move_made, score  = move(board)
        if move_made:
            return board, True, score
        move_order.pop(move_index)
    return board, False, score


cpdef add_new_tile(board):
    cdef int tile_value = NEW_TILE_DISTRIBUTION[np.random.randint(0, len(NEW_TILE_DISTRIBUTION))]
    cdef np.ndarray[long long, ndim=1] tile_row_options
    cdef np.ndarray[long long, ndim=1] tile_col_options
    tile_row_options, tile_col_options = np.nonzero(np.logical_not(board))
    cdef int tile_loc = np.random.randint(0, len(tile_row_options))
    board[tile_row_options[tile_loc], tile_col_options[tile_loc]] = tile_value
    return board


cpdef check_for_win(board):
    return 2048 in board