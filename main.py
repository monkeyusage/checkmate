import chess
import numpy


def piece_values(board: chess.Board, side: chess.Color) -> tuple[int, int]:
    # we want to maximize the attack area and minimize the number of unprotected pieces, also minimize opponent value
    own = sum([piece.piece_type for piece in board.piece_map().values() if piece.color == side])
    other = sum([piece.piece_type for piece in board.piece_map().values() if piece.color != side])
    return own, other


def get_board_value(board: chess.Board, side: chess.Color) -> float:
    # initial value is the sum of each player's pieces
    own, other = piece_values(board, side)
    for (square, piece) in board.piece_map().items():
        # if a piece is under attack remove half of its value from the piece owner and add it to the opponent
        multiplier = 1 if side == piece.color else -1 
        breakpoint()
        if board.attacks(square):
            own += (piece.piece_type / 2) * multiplier
            other += (piece.piece_type / 2) * multiplier
    return own - other


def play(board: chess.Board, side: chess.Color) -> chess.Move:
    best_value = -numpy.inf
    best_move = chess.Move.null()
    for move in board.legal_moves:
        board.push(move)  # play the move
        if board.is_checkmate():
            board.pop()
            return move 
        board_value = get_board_value(board, side)
        if board_value >= best_value:
            best_move = move
        board.pop()
    return best_move


def get_user_move(board: chess.Board) -> chess.Move:
    print('Available moves are:')
    print('\n'.join([move.uci() for move in board.legal_moves]))
    while True:
        try:
            move_str = input('> ')
            match move_str:
                case 'quit':
                    exit()
                case _:
                    return chess.Move.from_uci(move_str)
        except chess.InvalidMoveError:
            print('Invalid move, enter one of the legal moves')


def main() -> None:
    board = chess.Board()
    while board.outcome() is None:
        # debug, only use 'AI' for white player
        move = (
            play(board, board.turn)
            if board.turn == chess.WHITE
            else get_user_move(board)
        )
        board.push(move)
    outcome = board.outcome()
    assert outcome is not None
    print(f'Outcome of the game is: {outcome.winner if outcome.winner is not None else "no one"}')


if __name__ == '__main__':
    main()
