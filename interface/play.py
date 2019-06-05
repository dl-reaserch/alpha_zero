"""Interface for SmyGo self-play"""
from smyAlphaZero.go import GameState

class play_match(object):
    """Interface to handle play between two players."""
    def __init__(self,player1,player2,save_dir=None,size=9):
        self.player1 = player1
        self.player2 = player2
        self.state = GameState(size=size)
        # I Propose that GameState should take a top-level save directory,
        # then automatically generate the specific file name

    def play(self):
        """Play one turn, update game state, save to disk"""
        end_of_game = self._play(self.player1)
        return end_of_game


