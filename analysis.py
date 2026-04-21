import json
import numpy as np
from collections import defaultdict

SIZE = 15
flat = [64, 80, 66, 50, 95, 65, 35, 109, 51, 67, 36, 21, 34, 33, 81, 96, 19, 49, 38, 37, 52, 53, 69, 110, 84, 68, 54, 39, 70, 86, 124, 112, 111, 128, 144, 114, 100, 127, 55, 25, 11, 85, 97, 83, 113, 129, 130, 126, 125, 142, 158, 172, 157, 156, 170, 78, 94, 48, 63, 47, 46, 77, 76, 17, 1, 62, 32, 107, 92, 108, 106, 91, 24, 10, 116, 102, 132, 148, 160, 161, 145, 115, 146, 174, 175, 190, 143, 147, 131, 159, 133, 134, 3, 204, 188, 176, 162, 189, 219, 206, 178, 194, 203, 173, 171, 221, 191, 187, 185, 199, 215, 200, 2, 4, 153, 139, 169, 137, 168, 167, 123, 138, 136, 140, 141, 201, 198, 183, 122, 121, 118, 104, 56, 42, 41, 26, 27, 13, 58, 57, 103, 89, 88, 73, 101, 119, 74, 149]

players = [1 if i % 2 == 0 else -1 for i in range(len(flat))]
game_moves = list(zip(flat, players))

def rc(idx): 
    return idx // SIZE, idx % SIZE

def valid(r, c): 
    return 0 <= r < SIZE and 0 <= c < SIZE

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

def find_sequences(board, player):
    found = defaultdict(list)
    vis = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r, c] != player: 
                continue
            for dr, dc in DIRS:
                if (r, c, dr, dc) in vis: 
                    continue
                sr, sc = r, c
                while valid(sr - dr, sc - dc) and board[sr - dr, sc - dc] == player: 
                    sr -= dr
                    sc -= dc
                L = 0
                er, ec = sr, sc
                while valid(er, ec) and board[er, ec] == player:
                    vis.add((er, ec, dr, dc))
                    er += dr
                    ec += dc
                    L += 1
                if L < 2: 
                    continue
                br, bc = sr - dr, sc - dc
                ar, ac = er, ec
                ob = valid(br, bc) and board[br, bc] == 0
                oa = valid(ar, ac) and board[ar, ac] == 0
                found[L].append({
                    'start': (sr, sc),
                    'end': (er - dr, ec - dc),
                    'dir': (dr, dc),
                    'length': L,
                    'open_ends': int(ob) + int(oa)
                })
    return found

def count_threats(board, player):
    seqs = find_sequences(board, player)
    o4 = sum(1 for s in seqs.get(4, []) if s['open_ends'] >= 1)
    o3 = sum(1 for s in seqs.get(3, []) if s['open_ends'] == 2)
    wins = len(seqs.get(5, []))
    return o4, o3, wins

def has_five(board, player):
    seqs = find_sequences(board, player)
    return any(s['length'] >= 5 for s in seqs.get(5, []))

def winning_moves(board, player):
    result = []
    for idx in range(225):
        r, c = rc(idx)
        if board[r, c] != 0: 
            continue
        board[r, c] = player
        if has_five(board, player): 
            result.append(idx)
        board[r, c] = 0
    return result

def fork_squares(board, player):
    """Squares where player can create 2+ open-4s."""
    result = []
    for idx in range(225):
        r, c = rc(idx)
        if board[r, c] != 0: 
            continue
        board[r, c] = player
        o4, _, w = count_threats(board, player)
        if o4 + w >= 2: 
            result.append(idx)
        board[r, c] = 0
    return result

def double_threat_squares(board, player):
    """Squares creating 1 open-4 + 1+ open-3 (softer fork)."""
    result = []
    for idx in range(225):
        r, c = rc(idx)
        if board[r, c] != 0: 
            continue
        board[r, c] = player
        o4, o3, w = count_threats(board, player)
        if (o4 + w >= 1) and (o3 >= 1) and (o4 + w < 2): 
            result.append(idx)
        board[r, c] = 0
    return result

def capture_score(board, player):
    """Heuristic positional value: weighted sum of all sequences."""
    seqs = find_sequences(board, player)
    s = 0
    for L, w in {5: 1e6, 4: 1e4, 3: 5e2, 2: 50}.items():
        for run in seqs.get(L, []):
            s += w * (run['open_ends'] + 1)
    return s

def longest_threat_chain(records, player):
    """Longest run of consecutive moves by player that each created a threat."""
    pk = 'x' if player == 1 else 'o'
    max_chain = 0
    cur = 0
    for rec in records:
        if rec['player'] == pk:
            if rec['new_o4'] > 0 or rec['new_o3'] > 0 or rec['is_fork']:
                cur += 1
                max_chain = max(max_chain, cur)
            else:
                cur = 0
    return max_chain

def board_control_9regions(board):
    """Count X and O stones in each of 9 3x3 macro-regions."""
    regions = {}
    for ri in range(3):
        for ci in range(3):
            r0, c0 = ri * 5, ci * 5
            sub = board[r0:r0+5, c0:c0+5]
            regions[(ri, ci)] = {'x': int((sub == 1).sum()), 'o': int((sub == -1).sum())}
    return regions


# ── Main analysis ─────────────────────────────────────────────────────────────
board = np.zeros((SIZE, SIZE), dtype=int)
records = []
stats = {'x': defaultdict(int), 'o': defaultdict(int)}

# Tracking state for tempo / sente-gote analysis
prev_had_threat = {'x': False, 'o': False}

for seq_i, (move_flat, player) in enumerate(game_moves):
    r, c = rc(move_flat)
    pk = 'x' if player == 1 else 'o'
    opp = -player
    opk = 'o' if pk == 'x' else 'x'
    
    # ── Pre-move ───────────────────────────────────────────────────────────────
    pre_o4p, pre_o3p, pre_wp = count_threats(board, player)
    pre_o4o, pre_o3o, pre_wo = count_threats(board, opp)
    
    wins_avail = winning_moves(board, player)
    opp_wins_avail = winning_moves(board, opp)
    
    forks_avail = fork_squares(board, player)          # fork opportunities this player has
    opp_forks_avail = fork_squares(board, opp)         # opponent fork opportunities (threat to block)
    double_avail = double_threat_squares(board, player)
    
    # ── Place ─────────────────────────────────────────────────────────────────
    board[r, c] = player
    
    # ── Post-move ──────────────────────────────────────────────────────────────
    post_o4p, post_o3p, post_wp = count_threats(board, player)
    post_o4o, post_o3o, post_wo = count_threats(board, opp)
    is_win = has_five(board, player)
    
    # ── Metrics ────────────────────────────────────────────────────────────────
    new_o4 = max(0, post_o4p - pre_o4p)
    new_o3 = max(0, post_o3p - pre_o3p)
    blocked_4 = post_o4o < pre_o4o
    blocked_3 = post_o3o < pre_o3o
    is_fork = (post_o4p >= 2) or (is_win and post_o4p >= 1)  # 2+ threats simultaneously
    is_double_threat = (post_o4p >= 1 and post_o3p >= 1 and not is_fork)
    
    missed_win = (len(wins_avail) > 0 and move_flat not in wins_avail)
    opp_had_win = len(opp_wins_avail) > 0
    failed_block = opp_had_win and move_flat not in opp_wins_avail
    
    fork_opportunity_existed = len(forks_avail) > 0
    fork_taken = move_flat in forks_avail
    fork_missed = fork_opportunity_existed and not fork_taken and not missed_win  # had fork, didn't take it
    
    opp_fork_existed = len(opp_forks_avail) > 0
    opp_fork_blocked = opp_fork_existed and move_flat in opp_forks_avail
    
    # Sente (attacking) vs gote (reactive/defensive)
    is_sente = new_o4 > 0 or new_o3 > 0 or is_fork
    is_gote = blocked_4 or blocked_3
    
    # Threat maintained
    threat_maintained = post_o4p >= 1
    px_score = capture_score(board, 1)
    po_score = capture_score(board, -1)
    
    rec = {
        'n': seq_i + 1, 'player': pk, 'flat': move_flat, 'pos': (r, c),
        'is_win': is_win,
        'new_o4': new_o4, 'new_o3': new_o3,
        'total_o4': post_o4p, 'total_o3': post_o3p,
        'blocked_4': blocked_4, 'blocked_3': blocked_3,
        'is_fork': is_fork, 'is_double_threat': is_double_threat,
        'fork_opp_count': len(forks_avail),
        'fork_taken': fork_taken, 'fork_missed': fork_missed,
        'fork_opportunity_existed': fork_opportunity_existed,
        'opp_fork_existed': opp_fork_existed, 'opp_fork_blocked': opp_fork_blocked,
        'missed_win': missed_win, 'failed_block': failed_block,
        'is_sente': is_sente, 'is_gote': is_gote,
        'threat_maintained': threat_maintained,
        'pos_x': px_score, 'pos_o': po_score,
        'opp_had_win': opp_had_win,
        'wins_avail': len(wins_avail),
        'opp_wins_avail': len(opp_wins_avail),
    }
    records.append(rec)
    
    # Stats
    s = stats[pk]
    s['new_o4'] += new_o4
    s['new_o3'] += new_o3
    s['blocked_4'] += int(blocked_4)
    s['blocked_3'] += int(blocked_3)
    s['forks'] += int(is_fork)
    s['double_threats'] += int(is_double_threat)
    s['fork_opps'] += int(fork_opportunity_existed)
    s['fork_taken'] += int(fork_taken)
    s['fork_missed'] += int(fork_missed)
    s['opp_fork_existed'] += int(opp_fork_existed)
    s['opp_fork_blocked'] += int(opp_fork_blocked)
    s['missed_wins'] += int(missed_win)
    s['failed_blocks'] += int(failed_block)
    s['sente'] += int(is_sente)
    s['gote'] += int(is_gote)
    
    if is_win: 
        s['wins'] += 1
        print(f"WIN: {'X' if player == 1 else 'O'} on move {seq_i + 1} at ({r},{c})")
        break

n_played = len(records)
if not any(r['is_win'] for r in records):
    print(f"No winner in {n_played} moves")

# ── Additional metrics ─────────────────────────────────────────────────────────
for pk in ['x', 'o']:
    s = stats[pk]
    mv = [r for r in records if r['player'] == pk]
    n = len(mv)
    s['moves'] = n
    s['fork_capture_rate'] = s['fork_taken'] / max(s['fork_opps'], 1)
    s['defence_rate'] = (s['blocked_4'] + s['blocked_3']) / max(n, 1)
    s['attack_rate'] = (s['new_o4'] + s['new_o3']) / max(n, 1)
    s['sente_ratio'] = s['sente'] / max(n, 1)
    s['threat_efficiency'] = (s['new_o4'] * 2 + s['new_o3']) / max(n, 1)
    s['longest_threat_chain'] = longest_threat_chain(records, 1 if pk == 'x' else -1)
    
    forced = sum(1 for r in mv if r['is_gote'] and not r['is_sente'])
    s['purely_defensive'] = forced
    s['purely_defensive_rate'] = forced / max(n, 1)

print("\n=== GAME 4 — FULL STATISTICS ===")
for pk, pn in [('x', 'X'), ('o', 'O')]:
    s = stats[pk]
    n = s['moves']
    print(f"\n{'─'*55}")
    print(f"{pn}  ({n} moves)")
    print(f"  OFFENCE:")
    print(f"    Open-4s created:        {s['new_o4']:4d}   ({s['new_o4'] / n * 100:.1f}%/move)")
    print(f"    Open-3s created:        {s['new_o3']:4d}   ({s['new_o3'] / n * 100:.1f}%/move)")
    print(f"    Fork moves:             {s['forks']:4d}")
    print(f"    Double-threat moves:    {s['double_threats']:4d}")
    print(f"    Fork opp. existed:      {s['fork_opps']:4d}   times")
    print(f"    Fork captured:          {s['fork_taken']:4d}   / {s['fork_opps']}  ({s['fork_capture_rate'] * 100:.1f}%)")
    print(f"    Fork missed (not taken):{s['fork_missed']:4d}")
    print(f"    Threat efficiency:      {s['threat_efficiency']:.3f}  (weighted 4s×2)")
    print(f"    Longest threat chain:   {s['longest_threat_chain']:4d}  consecutive threat moves")
    print(f"    Sente ratio:            {s['sente_ratio'] * 100:.1f}%")
    print(f"  DEFENCE:")
    print(f"    Fours blocked:          {s['blocked_4']:4d}")
    print(f"    Threes blocked:         {s['blocked_3']:4d}")
    print(f"    Opp fork existed:       {s['opp_fork_existed']:4d}   times player faced opp fork")
    print(f"    Opp fork blocked:       {s['opp_fork_blocked']:4d}   ({s['opp_fork_blocked'] / max(s['opp_fork_existed'], 1) * 100:.1f}%)")
    print(f"    Purely defensive moves: {s['purely_defensive']:4d}   ({s['purely_defensive_rate'] * 100:.1f}%)")
    print(f"  ERRORS:")
    print(f"    Missed wins:            {s['missed_wins']:4d}")
    print(f"    Failed blocks:          {s['failed_blocks']:4d}")

# Critical events
print(f"\n{'='*60}")
print("CRITICAL EVENTS")
print(f"{'='*60}")
for r in records:
    notes = []
    pl = 'X' if r['player'] == 'x' else 'O'
    row, col = r['pos']
    if r['missed_win']: notes.append(f"⚠ MISSED WIN ({r['wins_avail']} wins available)")
    if r['failed_block']: notes.append(f"⚠ FAILED BLOCK (opp had {r['opp_wins_avail']} win(s))")
    if r['is_fork']: notes.append(f"★ FORK ({r['total_o4']} open-4s)")
    if r['is_double_threat']: notes.append(f"◆ double-threat")
    if r['fork_missed']: notes.append(f"✗ missed fork ({r['fork_opp_count']} fork sq available)")
    if r['new_o4'] > 0: notes.append(f"+{r['new_o4']} open-4")
    if r['new_o3'] > 0: notes.append(f"+{r['new_o3']} open-3")
    if r['blocked_4']: notes.append("BLOCKED-4")
    if r['opp_fork_blocked']: notes.append("BLOCKED OPP-FORK")
    if r['is_win']: notes.append("★★★ WON")
    
    if notes:
        print(f"  Move {r['n']:4d} {pl} ({row},{col}): {', '.join(notes)}")

# Board region analysis at game end
print(f"\n{'='*60}")
print("FINAL BOARD REGION CONTROL")
print(f"{'='*60}")
regions = board_control_9regions(board)
region_names = {0: 'TL', 1: 'TC', 2: 'TR', 3: 'ML', 4: 'MC', 5: 'MR', 6: 'BL', 7: 'BC', 8: 'BR'}
for ri in range(3):
    for ci in range(3):
        reg = regions[(ri, ci)]
        dom = 'X' if reg['x'] > reg['o'] else ('O' if reg['o'] > reg['x'] else '=')
        print(f"  {region_names[ri * 3 + ci]}: X={reg['x']} O={reg['o']} → {dom}")

# Phase analysis
print(f"\n{'='*60}")
print("PHASE BREAKDOWN")
print(f"{'='*60}")
for lo, hi, ph in [(1, min(30, n_played), 'Opening'), (31, min(80, n_played), 'Middle'), (81, min(130, n_played), 'Late mid'), (131, n_played, 'Endgame')]:
    if lo > n_played: 
        continue
    phase = [r for r in records if lo <= r['n'] <= hi]
    xm = [r for r in phase if r['player'] == 'x']
    om = [r for r in phase if r['player'] == 'o']
    print(f"\n  {ph} (moves {lo}–{hi}):")
    print(f"    X: {sum(r['new_o4'] for r in xm)} 4s, {sum(r['new_o3'] for r in xm)} 3s, "
          f"{sum(r['blocked_4'] + r['blocked_3'] for r in xm)} blocks, "
          f"{sum(1 for r in xm if r['is_fork'])} forks, "
          f"{sum(1 for r in xm if r['fork_missed'])} fork-missed")
    print(f"    O: {sum(r['new_o4'] for r in om)} 4s, {sum(r['new_o3'] for r in om)} 3s, "
          f"{sum(r['blocked_4'] + r['blocked_3'] for r in om)} blocks, "
          f"{sum(1 for r in om if r['is_fork'])} forks, "
          f"{sum(1 for r in om if r['fork_missed'])} fork-missed")

# Save
def cv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, defaultdict): return dict(o)
    raise TypeError(type(o))

with open('log/287th_148plies.json', 'w') as f:
    json.dump({
        'records': records,
        'stats_x': dict(stats['x']),
        'stats_o': dict(stats['o']),
        'total': n_played,
        'final_board': board.tolist()
    }, f, default=cv)

print("\nSaved game4.json")