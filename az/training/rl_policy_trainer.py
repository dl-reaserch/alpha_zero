import os
import re
from smyAlphaZero.ai import MCTSPlayer
import json
from smyAlphaZero import go
import az.preprocessing.preprocessing as pr
import numpy as np
from go_engine import GoBoard
import time


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
stone_radius = 10
IS_SHOW_STONES=False

if IS_SHOW_STONES:
    import pygame


# benchmarks/rl_agz_ckpt,benchmarks/weights,benchmarks/rl_output,'--learning-rate', '0.001', '--save-every', '2',
# '--game-batch', '20', '--iterations', '10', '--verbose'
def run_n_games(model_json,
                pvnet,
                game_batch,
                move_limit,
                resignation_threshold=0.05):
    print("game_batch:", game_batch, "--move_limit:", move_limit)
    with open(model_json, 'r') as f:
        object_specs = json.load(f)
    board_size = pvnet.board
    if IS_SHOW_STONES:
        go_engine = GoBoard(500, 500, board_size + 1, 30)
    batch_xyzs = []

    for game_batch_i in range(game_batch):
        player = MCTSPlayer(pvnet.estimate_value, pvnet.estimate_policy, "player")
        opponent = MCTSPlayer(pvnet.estimate_value, pvnet.estimate_policy, "opponent_player")
        print("game_batch_:",game_batch_i)
        pvnet.load_ckpts(file_path=object_specs['az_ckpt_path'])
        state = go.GameState(size=board_size)
        if IS_SHOW_STONES:
            clone_screen = go_engine.screen
            go_engine.init_engine(clone_screen)
            pygame.display.flip()

        xyzs = []
        z = np.zeros([3])

        player_color = go.BLACK if game_batch_i % 2 == 0 else go.WHITE
        current_player = player if state.get_current_player() == player_color else opponent
        s_t_planes = []

        s_t, move_matrix, move, max_leaf_value, s_t_planes = init_az_state(state,
                                                                           board_size,
                                                                           current_player,
                                                                           s_t_planes, 1)

        print(current_player.player_flag, "--执 黑", "--落子:", move)
        xyzs.append([s_t, move_matrix])

        _ = state.do_move(move)
        if IS_SHOW_STONES:
            show_stone(go_engine, clone_screen, state)

        new_s_t_planes = s_t_planes

        for step_i in range(1, move_limit):
            if IS_SHOW_STONES:
                for event in pygame.event.get():

                    # 检查是否关闭窗口
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        continue
            temperature = 1 if step_i <= 12 else 0.00001
            current_player = player if state.get_current_player() == player_color else opponent
            s_t, move_matrix, move, max_leaf_value, new_s_t_planes = get_az_state(state,
                                                                                  board_size,
                                                                                  current_player,
                                                                                  new_s_t_planes,
                                                                                  temperature,
                                                                                  player_color)
            color = "黑" if step_i % 2 ==0 else "白"
            print("game_batch_:",game_batch_i,"--step_:",step_i,"--",current_player.player_flag,"--执 ",color,"--落子:",move)

            xyzs.append([s_t, move_matrix])
            is_end_of_game = state.do_move(move)
            if IS_SHOW_STONES:
                show_stone(go_engine, clone_screen, state)
            if is_end_of_game:
                print("some player loss all stones, game over --get the winer:", state.get_winner())
            if step_i == move_limit - 1 or is_end_of_game:
                print( "move_limit game over --get the winer:", state.get_winner())
                if state.get_winner() == player_color:
                    z[0] = 1
                elif state.get_winner() == -player_color:
                    z[2] = 1
                else:
                    z[1] = 1
                for s in xyzs:
                    s.append(z)
                if IS_SHOW_STONES:
                    go_engine.reset_board(go_engine.screen)
                break
        batch_xyzs.append(xyzs)

    s_batch_xyzs = np.reshape(batch_xyzs, (-1, 3))
    np.random.shuffle(s_batch_xyzs)
    batch_size = 32
    n_batchs = len(s_batch_xyzs) // batch_size

    def get_next_batches():
        for batch_i in range(n_batchs + 1):
            start = batch_i * batch_size
            end = (batch_i + 1) * batch_size
            train_xyzs = s_batch_xyzs[start:end]
            yield list(zip(*train_xyzs))

    batch_gen = get_next_batches()
    index = 0
    for batch_i in range(n_batchs + 1):
        next_batches = [*next(batch_gen)]
        index += (batch_i + 1) * batch_size
        pvnet.process(next_batches, model_json, index)


def show_stone(go_engine, screen, state):
    boards = state.board
    points = {}
    for x_i, x in enumerate(boards):
        for y_i, y in enumerate(x):
            pos_x = (x_i + 1) * go_engine.grid_width
            pos_y = (y_i + 1) * go_engine.grid_width
            if y == 1:
                points[(pos_x, pos_y)] = BLACK
            elif y == -1:
                points[(pos_x, pos_y)] = WHITE
            else:
                pass

    go_engine.show_stones(screen, points, stone_radius)


def get_az_state(state, board_size, current_player, new_s_t_planes, temperature, player_color):
    s_t, new_s_t_planes = get_current_st(state, board_size, new_s_t_planes, current_player)
    max_leaf_value, move = do_mscts(current_player, state, temperature)
    move_matrix = trans2move_matrix(board_size, move)
    return s_t, move_matrix, move, max_leaf_value, new_s_t_planes


def init_az_state(state, board_size, current_player, s_t_planes, temperature):
    s_t, s_t_planes = init_st(state, board_size, s_t_planes)
    max_leaf_value, move = do_mscts(current_player, state, temperature)
    move_matrix = trans2move_matrix(board_size, move)
    return s_t, move_matrix, move, max_leaf_value, s_t_planes


def init_st(state, board_size, s_t_planes):
    init_planes = pr.get_board_new(state)
    for i in range(8):
        s_t_planes.append(init_planes[0])
        s_t_planes.append(init_planes[1])

    s_t = np.stack((init_planes[0], init_planes[1], init_planes[0], init_planes[1], init_planes[0], init_planes[1],
                    init_planes[0], init_planes[1],
                    init_planes[0], init_planes[1], init_planes[0], init_planes[1], init_planes[0], init_planes[1],
                    init_planes[0], init_planes[1], np.ones([board_size, board_size])),
                   axis=2)

    return s_t, s_t_planes


def get_current_st(state, board_size, new_s_t_planes, current_player):
    new_planes = pr.get_board_new(state)
    new_s_t_planes = new_s_t_planes[2:len(new_s_t_planes)]

    new_s_t_planes = np.append(new_s_t_planes, [new_planes[0]], axis=0)
    new_s_t_planes = np.append(new_s_t_planes, [new_planes[1]], axis=0)
    if state.get_current_player() == 1:

        s_t = np.append(new_s_t_planes, [np.ones([board_size, board_size])], axis=0)
    else:

        s_t = np.append(new_s_t_planes, [np.zeros([board_size, board_size])], axis=0)

    return np.stack(s_t, axis=2), new_s_t_planes


def do_mscts(player, state, temperature=1):
    moved = {}
    boards = state.board
    for x_i, x in enumerate(boards):
        for y_i, y in enumerate(x):
            if not y == 0:
                moved[(x_i, y_i)] = 1
    max_leaf_value, move = player.get_move(state, temperature, moved,{})
    return max_leaf_value, move


def trans2move_matrix(board_size, move):
    move_matrix = np.zeros([board_size * board_size + 1])
    move_index = board_size * board_size if move == go.PASS_MOVE else (move[0]) * board_size + move[1]
    move_matrix[move_index] = 1
    return move_matrix


def run_training(pvnet, cmd_line_args=None):
    print("run_training...")
    import argparse
    parser = argparse.ArgumentParser(
        description='Perform reinforcement learning to improve given policy network. Second phase of pipeline')
    # 策略模型存放的位置

    parser.add_argument("model_json", help='Path to policy model.')
    # 初始化权重文件的位置
    parser.add_argument("initial_weights",
                        help="Path to HDF5 file with inital weights (i.e. result of supervised training).")  # noqa: E501
    # 将要存放模型参数和元数据的位置
    parser.add_argument("out_directory",
                        help="Path to folder where the model params and metadata will be saved after each epoch.")  # noqa: E501
    parser.add_argument("--learning-rate", help="Keras learning rate (Default: 0.001)", type=float,
                        default=0.001)  # noqa: E501
    parser.add_argument("--policy-temp", help="Distribution temperature of players using policies (Default: 0.67)",
                        type=float, default=0.67)  # noqa: E501
    parser.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 500)", type=int,
                        default=500)  # noqa: E501
    parser.add_argument("--record-every", help="Save learner's weights every n batches (Default: 1)", type=int,
                        default=1)  # noqa: E501
    # mini-batch
    parser.add_argument("--game-batch", help="Number of games per mini-batch (Default: 20)", type=int,
                        default=20)  # noqa: E501
    parser.add_argument("--move-limit", help="Maximum number of moves per game", type=int,
                        default=go.size * go.size * 2)  # noqa: E501
    # batch
    parser.add_argument('--iter_start', type=int, default=0)
    parser.add_argument("--iterations", help="Number of training batches/iterations (Default: 10000)", type=int,
                        default=10000)  # noqa: E501
    parser.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False,
                        action="store_true")  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False,
                        action="store_true")  # noqa: E501
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.resume:
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without existing output directory")

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print("creating output directory {}".format(args.out_directory))
        os.makedirs(args.out_directory)

    if args.resume:
        # if resuming, we expect initial_weights to be just a
        # "weights.#####.hdf5" file, not a full path
        if not re.match(r"weights\.\d{5}\.hdf5", args.initial_weights):
            raise ValueError("Expected to resume from weights file with name 'weights.#####.hdf5'")
        args.initial_weights = os.path.join(args.out_directory, os.path.basename(args.initial_weights))

        if not os.path.exists(args.initial_weights):
            raise ValueError("Cannot resume; weights {} do not exist".format(args.initial_weights))
        elif args.verbose:
            print("Resuming with weights {}".format(args.initial_weights))
        player_weights = os.path.basename(args.initial_weights)
        # TODO
        # iter_start = 1 + int(player_weights[8:13])

    # Set initial conditions



    if args.verbose:
        print("created player and opponent with temperature {}".format(args.policy_temp))

    if not args.resume:
        metadata = {
            "model_file": args.model_json,
            "init_weights": args.initial_weights,
            "learning_rate": args.learning_rate,
            "temperature": args.policy_temp,
            "game_batch": args.game_batch,
            "win_ratio": {}  # map from player to tuple of (opponent, win ratio) Useful for
            # validating in lieu of 'accuracy/loss'
        }
    else:
        with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
    # Append args of current run to history of full command args.
    metadata["cmd_line_args"] = metadata.get("cmd_line_args", [])
    metadata["cmd_line_args"].append(vars(args))

    def save_metadata():
        with open(os.path.join(args.out_directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

    for i_iter in range(args.iter_start, args.iterations + 1):
        run_n_games(args.model_json, pvnet, args.game_batch, args.move_limit)


if __name__ == '__main__':
    run_training()
