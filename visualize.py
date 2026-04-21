import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Parse the moves
# move_text = """
# Selected move: 64
# Opponent move: 66
# Selected move: 80
# Opponent move: 96
# Selected move: 50
# Opponent move: 78
# Selected move: 65
# Opponent move: 35
# Selected move: 49
# Opponent move: 79
# Selected move: 48
# Opponent move: 47
# Selected move: 81
# Opponent move: 97
# Selected move: 33
# Opponent move: 17
# Selected move: 95
# Opponent move: 110
# Selected move: 62
# Opponent move: 109
# Selected move: 34
# Opponent move: 20
# Selected move: 19
# Opponent move: 4
# Selected move: 63
# Opponent move: 61
# Selected move: 32
# Opponent move: 16
# Selected move: 76
# Opponent move: 90
# Selected move: 18
# Opponent move: 3
# Selected move: 31
# Opponent move: 30
# Selected move: 52
# Opponent move: 51
# Selected move: 82
# Opponent move: 2
# Selected move: 83
# Opponent move: 84
# Selected move: 1
# Opponent move: 6
# Selected move: 5
# Opponent move: 93
# Selected move: 77
# Opponent move: 107
# Selected move: 108
# Opponent move: 121
# Selected move: 135
# Opponent move: 124
# Selected move: 139
# Opponent move: 122
# Selected move: 123
# Opponent move: 138
# Selected move: 152
# Opponent move: 106
# Selected move: 154
# Opponent move: 125
# Selected move: 91
# Opponent move: 141
# Selected move: 157
# Opponent move: 126
# Selected move: 105
# """

# moves = [int(line.split(": ")[1]) for line in move_text.strip().split("\n")]
moves = [160, 144, 130, 146, 143, 145, 147, 127, 161, 159, 131, 132, 163, 179, 133, 175, 148, 178, 118, 103, 162, 164, 176, 134, 149, 174, 129, 115, 190, 204, 189, 192, 206, 191, 157, 171, 173, 141, 155, 156, 126, 188, 172, 207, 223, 113, 85, 101, 112, 140, 201, 138, 139, 124, 108, 122, 152, 106, 154, 121, 123, 151, 136, 107, 184, 168, 169, 199, 185, 217, 153, 137, 77, 183, 167, 181, 93, 78, 125, 111, 109, 61, 128, 92, 64, 96, 82, 83, 66, 98, 67, 65, 97, 52, 53, 39, 187, 142, 186, 91, 76, 116, 202, 100, 102, 68, 84, 71, 55, 56, 86, 70, 72, 215, 205, 221, 218, 170, 88, 87, 119, 36, 193, 20, 4, 37, 38, 50, 35, 22, 23, 90, 8, 7, 49, 79, 95, 81, 63, 19, 21]
size = 15

fig, ax = plt.subplots(figsize=(8, 8))

def draw_static_board():
    ax.set_facecolor('#DCB35C')
    for i in range(size):
        ax.plot([0, size-1], [i, i], color='black', linewidth=1, zorder=1)
        ax.plot([i, i], [0, size-1], color='black', linewidth=1, zorder=1)
    ax.set_aspect('equal')
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels([chr(ord('A') + i) for i in range(size)])
    ax.set_yticklabels(range(1, size + 1))

def get_winning_stones(current_moves):
    # Determine the current player's moves
    is_black_turn = len(current_moves) % 2 == 1
    player_moves = [current_moves[i] for i in range(1 - int(is_black_turn), len(current_moves), 2)]
    
    board = set(player_moves)
    for m in player_moves:
        r, c = m // size, m % size
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            win_stones = [m]
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                nm = nr * size + nc
                if 0 <= nr < size and 0 <= nc < size and nm in board:
                    win_stones.append(nm)
                else:
                    break
            if len(win_stones) >= 5:
                # Return exactly 5 winning stones to highlight
                return win_stones[:5]
    return []

def update(frame):
    ax.clear()
    
    # Handle end of game delay frames
    actual_frame = min(frame, len(moves) - 1)
    
    draw_static_board()
    
    current_moves = moves[:actual_frame+1]
    
    # Check if this frame is a win
    winning_stones = get_winning_stones(current_moves)

    for i, move_idx in enumerate(current_moves):
        r = move_idx // size
        c = move_idx % size
        # Matrix row 0 is top, plotted at y = size-1
        y_pos = size - 1 - r
        x_pos = c
        
        color = 'black' if i % 2 == 0 else 'white'
        
        # Highlight most recent move or all winning stones
        if winning_stones and move_idx in winning_stones:
            edge_color = 'red'
            line_width = 3
        else:
            edge_color = 'red' if i == actual_frame else 'black'
            line_width = 2 if i == actual_frame else 1
            
        circle = plt.Circle((x_pos, y_pos), 0.4, color=color, ec=edge_color, lw=line_width, zorder=3)
        ax.add_patch(circle)
        
    ax.set_title(f"Gomoku Game History - Move {actual_frame + 1} ({'Black' if actual_frame % 2 == 0 else 'White'})")
    return []

# Create animation: 1 frame per second (interval = 1000ms)
# Add 2 extra frames at the end for a 2s delay
ani = animation.FuncAnimation(fig, update, frames=len(moves) + 4, interval=500, blit=False)

# Save the GIF
ani.save('log/gomoku_287th_141plies.gif', writer='pillow', fps=2)
plt.close()