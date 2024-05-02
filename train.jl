import Chess
import Flux


struct Cerberus{T}
  paths::T
end

Cerberus(paths...) = Cerberus(paths)
(model::Cerberus)(x::AbstractArray) = map(f -> f(x), model.paths)

Flux.@layer Cerberus


mutable struct Bot
  neuralnet::Flux.Chain
  wins::Int
end


Bot() = Bot(
  Flux.Chain(
    Flux.Conv((8, 8), 12 => 64, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.Conv((8, 8), 64 => 64, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.Conv((5, 5), 64 => 64, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.flatten,
    # output represents all possible moves in chess board for both white and black)
    Cerberus(
      Flux.Dense(64*64 => 64*64, Flux.sigmoid),
      Flux.Dense(64*64 => 1, Flux.tanh)
    )
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
  (policy, (board_value,)) = bot.neuralnet(to_array(board))
  # cancel impossible moves such as a1a1
  policy[[1:65:4096]...] .= 0
  # sort by highest probability
  moves = sortperm(vec(policy))
  max_index = length(moves)
  move_index = 1 
  selected_move = Chess.Move(moves[move_index])
  # iterate over moves stop at first highest ranked legal move
  legal_moves = Chess.moves(board)
  while !(selected_move in legal_moves)
    move_index += 1
    @inbounds selected_move = Chess.Move(moves[move_index])
    @assert move_index < max_index "legal_moves: $(legal_moves)"
  end
  selected_move
end

function play!(bot::Bot)
  game = Chess.Game()
  while !Chess.isterminal(game)
    board = game.node.board
    move = pick_move(bot, board)
    Chess.domove!(game, move)
  end
  if Chess.ischeckmate(game.node.board)
    bot.wins += 1
  end
end


function train()
  bot = Bot()
  while bot.wins == 0
    play!(bot, bot)
  end
end
