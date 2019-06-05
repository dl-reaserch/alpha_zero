import smyAlphaZero.go as go

state = go.GameState(size=9)
def test_eyes():
    state.do_move((1,0),1)
    state.do_move((4, 4),-1)
    state.do_move((0,1),1)
    state.do_move((5, 5),-1)
    state.do_move((5, 6),1)
    state.do_move((6, 6),-1)
    moves = [move for move in state.get_legal_moves(include_eyes=False)]
    #print(state.is_eye((0, 0), state.get_current_player()))
    print((0,0) in moves)



_children = {}
print(len(_children))
test_eyes()
