"""Connect Four, ported from Pgx (github.com/sotetsuk/pgx, pgx/connect_four.py).

Pgx writes the rules as array programs: the board is a flat `i32[42]`, a win
is found by gathering all 69 four-in-a-row index tuples from a table built at
import time, and the environment API (`core.Env.step`) is three `lax.cond`s
over lambdas, with `lax.select` for the rest. This port keeps the
observations, rewards, legal-action masks and the random draw of the first
player identical (checked in compare.py), and writes one game as one
`@expex.struct`, with plain Python:

* the board is `i32[Rows, Cols]`, and a move is `board[row, col] = self.color`;
* a win is counted by walking from the new stone with a `while` loop;
* `if self.terminated:` / `if won:` instead of `lax.cond(..., lambda: ...)`;
* a batch of games is `ConnectFour.new.vectorize(keys)` / `state.step.vectorize(actions)`.

    python examples/pgx/connect_four.py            # play random games, print one
    python -m expex examples/pgx/connect_four.py   # prove the shapes
"""

import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Batch, Bool, Dim, Key, f32, i32


class Rows(Dim, size=6): ...
class Cols(Dim, size=7): ...
class Players(Dim, size=2): ...


EMPTY = -1
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))  # down, right, and the two diagonals


@expex.struct
class ConnectFour:
    current_player: i32  # the player id (0 or 1) to move; not the colour
    observation: Bool[Rows, Cols, Players]  # [..., 0]: stones of the player to move
    rewards: f32[Players]
    terminated: Bool
    truncated: Bool
    legal_action_mask: Bool[Cols]
    step_count: i32
    color: i32  # the colour (0 or 1) to move: 0 moves first
    board: i32[Rows, Cols]  # EMPTY, 0 or 1; row 0 is the top
    winner: i32  # the winning colour, or -1

    @staticmethod
    def new(key: Key) -> "ConnectFour":
        """A new game; the player who moves first is drawn at random."""
        board = jnp.full((Rows, Cols), EMPTY, jnp.int32)
        game = ConnectFour(
            current_player=jnp.int32(jr.bernoulli(key)),
            observation=jnp.zeros((Rows, Cols, Players), bool),
            rewards=jnp.zeros(Players),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=jnp.ones(Cols, bool),
            step_count=jnp.int32(0),
            color=jnp.int32(0),
            board=board,
            winner=jnp.int32(-1),
        )
        return game.replace(observation=game.observe())

    def step(self, action: i32) -> "ConnectFour":
        """Pgx's `Env.step`: a finished game stays finished, an illegal move loses."""
        if self.terminated:
            game = self.replace(rewards=jnp.zeros(Players))
        else:
            game = self.replace(step_count=self.step_count + 1).drop(action)
        if not self.legal_action_mask[action]:
            rewards = jnp.ones(Players)
            rewards[self.current_player] = -1.0
            game = game.replace(rewards=rewards, terminated=jnp.bool_(True))
        if game.terminated:  # any action is "legal" once the game is over
            game = game.replace(legal_action_mask=jnp.ones(Cols, bool))
        return game.replace(observation=game.observe())

    def drop(self, col: i32) -> "ConnectFour":
        """Drop a stone of the colour to move into column `col`."""
        board = self.board
        row = Rows - 1 - (board[:, col] != EMPTY).sum()
        board[row, col] = self.color
        won = False
        for dr, dc in DIRECTIONS:
            won = won or line_length(board, row, col, dr, dc) >= 4
        rewards = jnp.zeros(Players)
        if won:
            rewards = -jnp.ones(Players)
            rewards[self.current_player] = 1.0  # the player who just moved
        legal = (board != EMPTY).sum(Rows) < Rows
        return self.replace(
            current_player=1 - self.current_player,
            color=1 - self.color,
            board=board,
            winner=self.color if won else -1,
            rewards=rewards,
            terminated=won or not legal.any(),
            legal_action_mask=legal,
        )

    def observe(self) -> Bool[Rows, Cols, Players]:
        """Two planes: the stones of the player to move, then the opponent's."""
        return jnp.stack([self.board == self.color, self.board == 1 - self.color], -1)


@expex.fn
def line_length(board: i32[Rows, Cols], row: i32, col: i32, dr: int, dc: int) -> i32:
    """How many stones of board[row, col]'s colour form a line through it along (dr, dc)."""
    n = 1
    for sign in (1, -1):
        r, c = row + sign * dr, col + sign * dc
        while 0 <= r < Rows and 0 <= c < Cols and board[r, c] == board[row, col]:
            n += 1
            r, c = r + sign * dr, c + sign * dc
    return n


@expex.fn(jit=True)
def random_games(key: Key, games: int, moves: int) -> tuple[Batch[ConnectFour, games], f32[games, Players]]:
    """`games` games after `moves` uniformly random legal moves each, and each player's total reward."""
    key, start = jr.split(key)
    state = ConnectFour.new.vectorize(jr.split(start, games))
    returns = jnp.zeros((games, Players))
    for _ in range(moves):
        key, pick = jr.split(key)
        action = jr.categorical(pick, jnp.log(state.legal_action_mask), axis=-1)
        state = state.step.vectorize(action)
        returns += state.rewards
    return state, returns


if __name__ == "__main__":
    state, returns = random_games(jr.key(0), 1000, 42)
    first = returns[:, 0]
    print(f"1000 random games: player 0 won {(first > 0).sum()}, lost {(first < 0).sum()}, drew {(first == 0).sum()}")
    print("game 0 at the end:")
    for row in state.board[0]:
        print(" ", " ".join(".xo"[int(v) + 1] for v in row))
    print("how drop() ran:", ConnectFour.drop.flow)
    print("how line_length() ran:", line_length.flow)
