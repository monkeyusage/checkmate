import Chess
import Flux
using Debugger


mutable struct Bot
  neuralnet::Flux.Chain
  wins::Int
end


Bot() = Bot(
  Flux.Chain(
    Flux.Conv((8, 8), 12 => 100, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.Conv((8, 8), 100 => 1000, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.Conv((8, 8), 1000 => 1000, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.flatten,
    Flux.Dense(64000 => 64*63*4, Flux.relu),
    # output represents all possible moves in chess board for both white and black)
    Flux.Dense(64*63*4 => 64*63*2, Flux.sigmoid),
  ), 0
)

function to_array(board::Chess.Board)::Array{Float16, 4}
  array = zeros(Float16, (8, 8, 12))
  # save board for each piece/color in an array
  for (index, (color, piece)) in enumerate(Iterators.product(
     [Chess.WHITE, Chess.BLACK], 
     [Chess.QUEEN, Chess.KING, Chess.BISHOP, Chess.KNIGHT, Chess.PAWN, Chess.ROOK])
  )
    array[:, :, index] = Chess.toarray(Chess.pieces(board, color, piece), Float16)
  end
  # reshapes for batshsize == 1
  reshape(array, (8, 8, 12, 1))
end

function pick_move(bot::Bot, board::Chess.Board)::Chess.Move
  # get the highest ranked legal prediction
  predictions = bot.neuralnet(to_array(board))
  moves = sortperm(vec(predictions))
  max_index = length(moves)
  move_index = 0
  selected_move = Chess.Move(moves[move_index])
  while !(selected_move in Chess.moves(board))
    move_index += 1
    selected_move = Chess.Move(moves[move_index])
    @assert move_index < max_index
  end
  print("Max searched moves $(move_index)")
  selected_move
end

function play!(whites::Bot, blacks::Bot)::Nothing
  game = Chess.Game()
  while !Chess.isterminal(game)
    board = game.node.board
    move = Chess.sidetomove(board) == Chess.WHITE ? pick_move(whites, board) : pick_move(blacks, board)
    Chess.domove!(game, move)
    print(board)
    break
  end
  if Chess.isstalemate(board)
    return nothing
  end
  if Chess.ischeckmate(board)
    winner = Chess.sidetomove(board) == Chess.WHITE ? blacks : whites
    winner.wins += 1
  end
end


function test()
  bot = Bot()
  board = Chess.@startboard
  play!(bot, bot)
end
