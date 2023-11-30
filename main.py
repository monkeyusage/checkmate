from dataclasses import dataclass, field
import random
import time
import heapq
import math
import sys
import io

import chess
import chess.svg
import cairosvg
import pygame


def get_board_value(board: chess.Board, side: chess.Color) -> float:
    pieces = board.piece_map().values()
    if board.is_checkmate():
        out = 3 
    elif board.is_check():
        out = 2
    else:
        own = sum([piece.piece_type for piece in pieces if piece.color == side])
        other = sum([piece.piece_type for piece in pieces if piece.color != side])
        out = own / other
    return out if side == board.turn else -out


@dataclass(order=True)
class Node:
    value: float
    turn: chess.Color = field(compare=False)
    move: chess.Move = field(compare=False)
    next_nodes: dict[chess.Move, 'Node'] = field(default_factory=dict, compare=False)
    queue: list['Node'] = field(default_factory=list, compare=False)

    def add_next(self, node: 'Node') -> None:
        self.next_nodes[node.move] = node
        heapq.heappush(self.queue, node)

    def __repr__(self) -> str:
        return f'Node(move:{self.move}, turn:{self.turn}, value:{self.value})'

    def size(self, acc: int = 0) -> int:
        if not self.next_nodes:
            return acc
        for next_node in self.next_nodes.values():
            acc += next_node.size(len([*next_node.next_nodes.values()]))
        return acc 


def play(board: chess.Board, side: chess.Color, max_depth: int, depth: int, prev_node: Node) -> chess.Move:
    """
    this function recursively fills the move tree
    when going down the leaves back to the root is 'optimizes' the node s values depending on the turn
    """
    start = time.perf_counter()
    if depth == max_depth:
        return chess.Move.null()

    for move in board.legal_moves:
        # if move has not yet been evaluated add it to the tree
        board.push(move)
        if (node:= prev_node.next_nodes.get(move)) is None:
            node_value = get_board_value(board, side)
            node = Node(turn=board.turn, move=move, value=node_value)
            prev_node.add_next(node)
        # otherwise do not recompute
        play(board, side, max_depth, depth+1, prev_node=node)
        board.pop()

    if depth != 0:
        # since we went through all next moves we can now optimize the node
        if prev_node.next_nodes:
            (best_node, *_) = heapq.nlargest(1, prev_node.queue)
            prev_node.value = best_node.value
        return chess.Move.null()

    # if we have equal moves we chose one randomly
    stop = time.perf_counter()
    max_value = max([node.value for node in prev_node.next_nodes.values()])
    print(f"Evaluated {prev_node.size()} nodes in {stop-start} seconds")
    return random.choice([node for node in prev_node.next_nodes.values() if node.value == max_value]).move


def get_user_move(board: chess.Board) -> chess.Move:
    while True:
        move_str = input('> ')
        if move_str == 'quit': exit()
        try:
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                raise chess.InvalidMoveError()
            return move
        except chess.InvalidMoveError:
            print('Invalid move, enter one of the legal moves')


def to_screen(board: chess.Board, size: int) -> pygame.Surface:
    buffer = chess.svg.board(board=board, size=size).encode('utf-8')
    png_buffer: bytes = cairosvg.svg2png(bytestring=buffer, output_height=size, output_width=size)  # type: ignore
    return pygame.image.load(io.BytesIO(png_buffer))


def get_square(xpos: int, ypos: int, size: int) -> chess.Square:
    letter = 'abcdefgh'[math.ceil((xpos/size) * 8) - 1]
    number = '12345678'[math.ceil((ypos/size) * 8) - 1]
    return chess.parse_square(letter + number)


def main() -> None:
    pygame.init()
    size = 420
    screen = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()
    board = chess.Board()
    root = Node(0, board.turn, chess.Move.null())
    while board.outcome() is None:
        screen_board = to_screen(board, size)
        screen.blit(screen_board, (0, 0))
        pygame.display.flip()

        def get_user_move() -> chess.Move:
            move_from: chess.Square | None = None
            move = chess.Move.null()
            while True:
                pygame.display.flip()
                events = pygame.event.get(eventtype=pygame.MOUSEBUTTONDOWN)
                quit_game = pygame.event.get(eventtype=pygame.QUIT)
                if quit_game:
                    pygame.quit()
                    sys.exit()
                if not events:
                    clock.tick(60)
                    continue
                event, *_ = events
                xpos, ypos = event.pos
                if move_from is None:
                    move_from = get_square(xpos, ypos, size)
                    continue
                move_to = get_square(xpos, ypos, size)
                move = chess.Move(move_from, move_to)
                if move not in board.legal_moves:
                    move_from = None
                    continue
                break
            
            breakpoint()
            return move

        move = play(board, board.turn, 2, 0, root) if board.turn == chess.WHITE else get_user_move()
        board.push(move)
        root = root.next_nodes[move]

    outcome = board.outcome()
    assert outcome is not None
    print(f'Outcome of the game is: {outcome.winner if outcome.winner is not None else "no one"}')
    pygame.quit()


if __name__ == '__main__':
    main()
