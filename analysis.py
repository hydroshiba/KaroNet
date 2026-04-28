import json
import os
import numpy as np

SIZE = 15

class FastGomokuAnalyzer:
    def __init__(self, moves):
        self.moves = moves
        self.board_np = np.full(225, -1, dtype=np.int8)
        self.records = []
        self.stats = {'x': {'new_o4': 0, 'new_o3': 0, 'blocked_4': 0, 'forks': 0,
                            'fork_opps': 0, 'fork_taken': 0, 'fork_missed': 0,
                            'opp_fork_existed': 0, 'opp_fork_blocked': 0,
                            'missed_wins': 0, 'failed_blocks': 0,
                            'sente': 0, 'purely_defensive': 0},
                      'o': {'new_o4': 0, 'new_o3': 0, 'blocked_4': 0, 'forks': 0,
                            'fork_opps': 0, 'fork_taken': 0, 'fork_missed': 0,
                            'opp_fork_existed': 0, 'opp_fork_blocked': 0,
                            'missed_wins': 0, 'failed_blocks': 0,
                            'sente': 0, 'purely_defensive': 0}}

        # Precompute all 572 possible winning lines of 5
        lines = []
        for r in range(SIZE):
            for c in range(SIZE - 4):
                lines.append([r * SIZE + c + i for i in range(5)])
        for c in range(SIZE):
            for r in range(SIZE - 4):
                lines.append([(r + i) * SIZE + c for i in range(5)])
        for r in range(SIZE - 4):
            for c in range(SIZE - 4):
                lines.append([(r + i) * SIZE + c + i for i in range(5)])
        for r in range(SIZE - 4):
            for c in range(4, SIZE):
                lines.append([(r + i) * SIZE + c - i for i in range(5)])
        self.lines5 = np.array(lines, dtype=np.int32)

    def rc(self, idx):
        return idx // SIZE, idx % SIZE

    def get_wins(self, player, board=None):
        """Squares that would complete a 5-in-a-row for player."""
        if board is None:
            board = self.board_np
        b5 = board[self.lines5]
        p_mask = (b5 == player)
        e_mask = (b5 == -1)
        valid = (p_mask.sum(axis=1) == 4) & (e_mask.sum(axis=1) == 1)
        # For each valid line, find the single empty square
        valid_e = e_mask & valid[:, np.newaxis]
        win_sq = np.where(valid_e, self.lines5, -1).max(axis=1)
        wins = np.unique(win_sq)
        return wins[wins != -1]

    def _count_per_board(self, b5, player):
        """Helper: count unique winning-threat squares per board in a batch.
        b5 shape: (E, n_lines, 5). Returns array of length E."""
        p_mask = (b5 == player)
        e_mask = (b5 == -1)
        valid = (p_mask.sum(axis=2) == 4) & (e_mask.sum(axis=2) == 1)
        valid_e = e_mask & valid[:, :, np.newaxis]
        win_sq = np.where(valid_e, self.lines5, -1).max(axis=2)
        win_sq.sort(axis=1)
        first = (win_sq[:, 0] != -1).astype(np.int32)
        diffs = (win_sq[:, 1:] != win_sq[:, :-1]) & (win_sq[:, 1:] != -1)
        return first + diffs.sum(axis=1)

    def batch_threat_counts(self, player):
        """For every empty square, return (indices, win_count, is_5inarow)."""
        empty = np.where(self.board_np == -1)[0]
        E = len(empty)
        if E == 0:
            return empty, np.array([], dtype=np.int32), np.array([], dtype=bool)

        batch = np.tile(self.board_np, (E, 1))
        batch[np.arange(E), empty] = player
        b5 = batch[:, self.lines5]
        is_win = np.any((b5 == player).sum(axis=2) == 5, axis=1)
        counts = self._count_per_board(b5, player)
        return empty, counts, is_win

    def get_o3_makers(self, player, board=None):
        """Squares that create >=2 winning threats for player (O3-makers)."""
        if board is None:
            board = self.board_np
        empty = np.where(board == -1)[0]
        if len(empty) == 0:
            return np.array([], dtype=np.int32)
        E = len(empty)
        batch = np.tile(board, (E, 1))
        batch[np.arange(E), empty] = player
        b5 = batch[:, self.lines5]
        counts = self._count_per_board(b5, player)
        return empty[counts >= 2]

    def _is_true_33(self, player):
        """After placing at self._cand, check if it's a true 3-3 fork."""
        opp = 1 - player
        empty = np.where(self.board_np == -1)[0]
        for b in empty:
            self.board_np[b] = opp
            makers = self.get_o3_makers(player)
            self.board_np[b] = -1
            if len(makers) == 0:
                return False
        return True

    def evaluate_move(self, player, move):
        """Evaluate a move. Returns dict with type and fork flags (true threats only)."""
        self.board_np[move] = player
        opp = 1 - player
        res = {"type": "none", "is_44": False, "is_43": False, "is_33": False}

        b5 = self.board_np[self.lines5]
        if np.any((b5 == player).sum(axis=1) == 5):
            res["type"] = "win"
            self.board_np[move] = -1
            return res

        wins_after = self.get_wins(player)
        nw = len(wins_after)

        if nw >= 2:
            res.update({"type": "u4", "is_44": True})
        elif nw == 1:
            forced = wins_after[0]
            self.board_np[forced] = opp
            o3 = len(self.get_o3_makers(player))
            self.board_np[forced] = -1
            res["type"] = "u4"
            if o3 > 0:
                res["is_43"] = True
        else:
            o3_makers = self.get_o3_makers(player)
            if len(o3_makers) >= 2:
                is_true = self._is_true_33(player)
                res["type"] = "o3"
                if is_true:
                    res["is_33"] = True
            elif len(o3_makers) == 1:
                res["type"] = "o3"

        self.board_np[move] = -1
        return res

    def _threat_lines_mask(self, player):
        """Return bool[225]: empty squares that lie on >= 2 lines with 3+ player stones.
        These are the only squares that could possibly create a fork."""
        b5 = self.board_np[self.lines5]
        p_count = (b5 == player).sum(axis=1)
        # Lines with 3+ player stones and at least 1 empty
        candidate_lines = (p_count >= 3) & ((b5 == -1).sum(axis=1) >= 1)
        counts = np.zeros(225, dtype=np.int8)
        for li in np.where(candidate_lines)[0]:
            for sq in self.lines5[li]:
                if self.board_np[sq] == -1:
                    counts[sq] += 1
        return counts >= 2

    def _candidate_mask(self):
        """Bool[225]: empty squares within distance 2 of any occupied stone."""
        occupied = np.where(self.board_np != -1)[0]
        board2d = self.board_np.reshape(SIZE, SIZE)
        mask = np.zeros(225, dtype=bool)
        for idx in occupied:
            r, c = idx // SIZE, idx % SIZE
            r0, c0 = max(0, r - 2), max(0, c - 2)
            r1, c1 = min(SIZE, r + 3), min(SIZE, c + 3)
            window = board2d[r0:r1, c0:c1]
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    if window[nr - r0, nc - c0] == -1:
                        mask[nr * SIZE + nc] = True
        return mask

    def board_control_9regions(self):
        regions = {}
        board2d = self.board_np.reshape(SIZE, SIZE)
        for ri in range(3):
            for ci in range(3):
                r0, c0 = ri * 5, ci * 5
                reg = board2d[r0:r0 + 5, c0:c0 + 5]
                regions[(ri, ci)] = {'x': int(np.sum(reg == 0)), 'o': int(np.sum(reg == 1))}
        return regions

    def longest_threat_chain(self, p_str):
        max_c = cur = 0
        for r in self.records:
            if r['player'] == p_str:
                if r['new_o4'] > 0 or r['new_o3'] > 0 or r['is_fork']:
                    cur += 1
                    if cur > max_c:
                        max_c = cur
                else:
                    cur = 0
        return max_c

    def vcf_chain_length(self, p_str):
        """Longest VCF chain: player creates a u4, opponent is forced to block
        that exact square, player creates another u4, etc.  Only counts
        consecutive new_o4 moves where the intervening opponent move was a
        forced block (blocked_4)."""
        recs = self.records
        max_chain = cur = 0

        for i, r in enumerate(recs):
            if r['player'] != p_str:
                continue
            if r['new_o4'] > 0:
                if cur == 0:
                    cur = 1
                else:
                    # Previous opponent move should have been a block
                    prev = recs[i - 1]
                    if prev['blocked_4']:
                        cur += 1
                    else:
                        cur = 1
                if cur > max_chain:
                    max_chain = cur
            else:
                cur = 0
        return max_chain

    def longest_forcing_sequence(self, p_str):
        """Longest sequence where the player's sente moves force opponent into
        purely defensive (gote) responses.  Counts consecutive turns where
        the player creates a threat AND the opponent's next move is gote
        or blocks a 4."""
        recs = self.records
        n = len(recs)
        max_seq = cur = 0
        i = 0

        while i < n:
            r = recs[i]
            if r['player'] != p_str:
                i += 1
                continue
            if r['is_sente'] and not r['is_win']:
                # Check if opponent's very next move is gote/blocked
                opp_move = recs[i + 1] if i + 1 < n else None
                if opp_move and opp_move['player'] != p_str:
                    if opp_move['is_gote'] or opp_move['blocked_4']:
                        cur += 1
                        if cur > max_seq:
                            max_seq = cur
                        i += 2  # Skip the opponent's move
                        continue
                # Chain broken
                cur = 0
            else:
                cur = 0
            i += 1
        return max_seq

    def longest_defensive_stand(self, p_str):
        """Longest sequence of consecutive blocks (forced-4 blocks or fork
        blocks) without committing a blunder."""
        max_stand = cur = 0
        for r in self.records:
            if r['player'] != p_str:
                continue
            if r['blocked_4'] or r['opp_fork_blocked']:
                cur += 1
                if cur > max_stand:
                    max_stand = cur
            elif r['failed_block']:
                cur = 0  # blunder resets
            # Non-block moves don't reset (player can't block if no threat)
        return max_stand

    def analyze(self):
        for turn, move in enumerate(self.moves):
            player = turn % 2
            opp = 1 - player
            p_str = 'x' if player == 0 else 'o'
            r, c = self.rc(move)

            # --- Pre-move threat state ---
            pre_my_wins = self.get_wins(player)
            pre_opp_wins = self.get_wins(opp)

            missed_win = len(pre_my_wins) > 0 and move not in pre_my_wins
            failed_block = (len(pre_opp_wins) > 0 and move not in pre_opp_wins
                            and move not in pre_my_wins)

            # --- Fork detection: batch win-counts, then per-candidate only for rare cases ---
            cand_mask = self._candidate_mask()
            cand_idx = np.where(cand_mask)[0]
            my_threat_mask = self._threat_lines_mask(player)
            opp_threat_mask = self._threat_lines_mask(opp)

            my_forks_avail = []
            opp_forks_avail = []

            if len(cand_idx) > 0:
                # Batch win-count for candidate squares
                empty_idx, my_wc, my_5row = self.batch_threat_counts(player)
                _, opp_wc, opp_5row = self.batch_threat_counts(opp)

                # Build lookup: empty square flat -> batch index
                idx_to_pos = {idx: i for i, idx in enumerate(empty_idx)}

                for idx in cand_idx:
                    i = idx_to_pos.get(idx)
                    if i is None:
                        continue

                    # --- Player forks ---
                    if not my_5row[i]:
                        wc = my_wc[i]
                        if wc >= 2:
                            my_forks_avail.append(idx)
                        elif wc == 1:
                            ev = self.evaluate_move(player, idx)
                            if ev['is_43']:
                                my_forks_avail.append(idx)
                        elif wc == 0 and my_threat_mask[idx]:
                            ev = self.evaluate_move(player, idx)
                            if ev['is_33']:
                                my_forks_avail.append(idx)

                    # --- Opponent forks ---
                    if not opp_5row[i]:
                        wc = opp_wc[i]
                        if wc >= 2:
                            opp_forks_avail.append(idx)
                        elif wc == 1:
                            ev = self.evaluate_move(opp, idx)
                            if ev['is_43']:
                                opp_forks_avail.append(idx)
                        elif wc == 0 and opp_threat_mask[idx]:
                            ev = self.evaluate_move(opp, idx)
                            if ev['is_33']:
                                opp_forks_avail.append(idx)

            fork_taken = move in my_forks_avail
            opp_fork_blocked = len(opp_forks_avail) > 0 and move in opp_forks_avail

            # --- Evaluate actual move ---
            ev = self.evaluate_move(player, move)

            # A move is only a "missed fork" if the player passed up a fork for
            # a non-winning move, with no forced defence and no direct win available.
            fork_missed = (len(my_forks_avail) > 0 and not fork_taken
                           and not missed_win and len(pre_opp_wins) == 0
                           and ev['type'] != 'win')

            # --- Apply move, check defence ---
            self.board_np[move] = player
            post_opp_wins = self.get_wins(opp)
            blocked_4 = len(pre_opp_wins) > 0 and len(post_opp_wins) == 0

            is_sente = (ev['type'] in ('u4', 'o3')) or ev['is_44'] or ev['is_43'] or ev['is_33']
            is_gote = blocked_4 and not is_sente

            rec = {
                'n': turn + 1, 'player': p_str, 'flat': move, 'pos': (r, c),
                'is_win': ev['type'] == 'win',
                'new_o4': 1 if ev['type'] == 'u4' else 0,
                'new_o3': 1 if ev['type'] == 'o3' else 0,
                'is_fork': ev['is_44'] or ev['is_43'] or ev['is_33'],
                'is_44': ev['is_44'], 'is_43': ev['is_43'], 'is_33': ev['is_33'],
                'blocked_4': blocked_4,
                'fork_opp_count': len(my_forks_avail),
                'fork_taken': fork_taken, 'fork_missed': fork_missed,
                'opp_fork_existed': len(opp_forks_avail) > 0,
                'opp_fork_blocked': opp_fork_blocked,
                'missed_win': missed_win, 'failed_block': failed_block,
                'is_sente': is_sente, 'is_gote': is_gote,
                'wins_avail': len(pre_my_wins),
                'opp_wins_avail': len(pre_opp_wins),
            }
            self.records.append(rec)

            s = self.stats[p_str]
            s['new_o4'] += rec['new_o4']
            s['new_o3'] += rec['new_o3']
            s['blocked_4'] += int(blocked_4)
            s['forks'] += int(rec['is_fork'])
            s['fork_opps'] += int(len(my_forks_avail) > 0)
            s['fork_taken'] += int(fork_taken)
            s['fork_missed'] += int(fork_missed)
            s['opp_fork_existed'] += int(rec['opp_fork_existed'])
            s['opp_fork_blocked'] += int(opp_fork_blocked)
            s['missed_wins'] += int(missed_win)
            s['failed_blocks'] += int(failed_block)
            s['sente'] += int(is_sente)
            s['purely_defensive'] += int(is_gote)

            if ev['type'] == 'win':
                break


def _phase_events(recs, lo, hi):
    """Collect key events for a phase range."""
    events = []
    for r in recs:
        if r['n'] < lo or r['n'] > hi:
            continue
        pl = r['player'].upper()
        if r['is_44']:
            events.append(f"Move {r['n']} ({pl}): unleashed a 4-4 fork")
        elif r['is_43']:
            events.append(f"Move {r['n']} ({pl}): sprung a 4-3 fork")
        elif r['is_33']:
            events.append(f"Move {r['n']} ({pl}): created a 3-3 fork")
        if r['missed_win']:
            events.append(f"Move {r['n']} ({pl}): missed a winning move")
        if r['failed_block']:
            events.append(f"Move {r['n']} ({pl}): failed to block opponent's win")
        if r['fork_missed']:
            events.append(f"Move {r['n']} ({pl}): missed a fork opportunity")
        if r['is_win']:
            events.append(f"Move {r['n']} ({pl}): GAME OVER — five in a row")
        if r['blocked_4'] and r['is_fork']:
            events.append(f"Move {r['n']} ({pl}): blocked a 4 AND counter-forked")
    return events


def _player_assessment(analyzer, p_str, p_name, won):
    """Generate a natural-language performance assessment for a player."""
    s = analyzer.stats[p_str]
    lines = []

    blunders = s['missed_wins'] + s['failed_blocks']
    fork_missed = s['fork_missed']
    fork_opps = s['fork_opps']
    missed_fork_pct = fork_missed / max(fork_opps, 1) * 100
    sente_pct = s['sente_ratio'] * 100
    chain = s['longest_threat_chain']
    opp_fork_block_rate = s['opp_fork_blocked'] / max(s['opp_fork_existed'], 1)

    vcf_chain = analyzer.vcf_chain_length(p_str)
    forcing_seq = analyzer.longest_forcing_sequence(p_str)
    def_stand = analyzer.longest_defensive_stand(p_str)

    # --- Grade (0 = best, 6 = worst) ---
    grades = ['excellently', 'strongly', 'well', 'above average',
              'averagely', 'below average', 'like a newcomer']

    if (won and blunders == 0 and fork_missed == 0
            and sente_pct >= 50 and s['forks'] >= 1 and chain >= 5):
        gi, detail = 0, 'controlled the game from start to finish'
    elif won and blunders == 0 and sente_pct >= 45:
        gi, detail = 1, 'capitalised on key moments and closed out the win'
    elif won and blunders <= 1:
        gi, detail = 2, 'managed the game well despite a few slips'
    elif (not won and blunders == 0 and sente_pct >= 50
          and fork_missed <= 1 and chain >= 4):
        gi, detail = 2, 'maintained strong pressure but could not land the final blow'
    elif (not won and blunders == 0 and def_stand >= 15
          and forcing_seq >= 2 and s['blocked_4'] >= 8):
        gi, detail = 2, ('played a resilient defensive game, holding off the opponent '
                         'until finally broken by an unblockable sequence')
    elif blunders <= 1 and fork_missed <= 1 and sente_pct >= 40:
        gi, detail = 3, 'showed solid play with a few missed chances'
    elif (blunders <= 1 and def_stand >= 10
          and s['blocked_4'] >= 5):
        gi, detail = 3, 'defended stubbornly but lacked the offensive firepower to turn the tide'
    elif blunders <= 1 and sente_pct >= 35:
        gi, detail = 4, 'had moments of threat but also some lapses'
    elif blunders <= 2 and sente_pct >= 25:
        gi, detail = 4, 'mixed performance with moments of threat and lapses'
    elif blunders <= 3:
        gi, detail = 5, 'struggled to sustain pressure and made costly errors'
    else:
        gi, detail = 6, 'multiple critical blunders and passive play'

    # Blunder caps: any blunder → max "well"; >1 blunders → max "above average"
    if blunders > 1:
        gi = max(gi, grades.index('above average'))
    elif blunders >= 1:
        gi = max(gi, grades.index('well'))

    lines.append(f'{p_name} played {grades[gi]} — {detail}.')

    # --- Offensive remarks ---
    off_notes = []
    if s['forks'] >= 2:
        off_notes.append(f'executed {s["forks"]} fork attacks')
    elif s['forks'] == 1:
        off_notes.append('executed a fork attack')
    if vcf_chain >= 3:
        off_notes.append(f'built a VCF chain of {vcf_chain} consecutive forcing moves')
    if chain >= 6:
        off_notes.append(f'sustained a threat chain of {chain} consecutive sente moves')
    elif chain >= 4:
        off_notes.append(f'mounted a threat chain of {chain} consecutive sente moves')
    if forcing_seq >= 5:
        off_notes.append(f'kept the opponent in gote for {forcing_seq} consecutive exchanges')
    if fork_opps >= 3 and missed_fork_pct <= 20:
        off_notes.append(f'found and took {s["fork_taken"]}/{fork_opps} fork chances')
    elif fork_missed >= 2:
        off_notes.append(f'squandered {fork_missed}/{fork_opps} fork opportunities')
    elif fork_missed == 1 and fork_opps >= 2:
        off_notes.append(f'missed {fork_missed} of {fork_opps} fork chances')
    if s['new_o4'] >= 10:
        off_notes.append(f'generated {s["new_o4"]} open-4 threats')
    if sente_pct >= 55:
        off_notes.append(f'held the initiative {sente_pct:.0f}% of the time')
    elif sente_pct < 35:
        off_notes.append(f'was on the back foot with only {sente_pct:.0f}% attacking moves')

    # --- Defensive remarks ---
    def_notes = []
    if def_stand >= 15:
        def_notes.append(f'held a defensive wall for {def_stand} consecutive blocks')
    elif def_stand >= 8:
        def_notes.append(f'stood firm through {def_stand} consecutive defensive stands')
    if s['blocked_4'] >= 10:
        def_notes.append(f'absorbed heavy pressure with {s["blocked_4"]} forced blocks')
    elif s['blocked_4'] >= 3:
        def_notes.append(f'handled {s["blocked_4"]} forced defensive blocks')
    if s['opp_fork_existed'] >= 3:
        if opp_fork_block_rate >= 0.6:
            def_notes.append(f'defused {s["opp_fork_blocked"]}/{s["opp_fork_existed"]} opponent fork threats')
        elif opp_fork_block_rate < 0.3:
            def_notes.append(f'struggled against opponent forks, blocking only {opp_fork_block_rate*100:.0f}%')
    if s['purely_defensive_rate'] > 0.25:
        def_notes.append(f'was forced into pure defence {s["purely_defensive_rate"]*100:.0f}% of the time')

    # --- Error remarks ---
    err_notes = []
    if s['failed_blocks'] >= 1:
        label = 'a critical blunder' if s['failed_blocks'] == 1 else f'{s["failed_blocks"]} critical blunders'
        err_notes.append(f'committed {label} by failing to block opponent wins')
    if s['missed_wins'] >= 1:
        label = 'a direct winning move' if s['missed_wins'] == 1 else f'{s["missed_wins"]} winning moves'
        err_notes.append(f'overlooked {label}')
    if fork_missed >= 1:
        err_notes.append(f'missed {fork_missed} fork {"opportunity" if fork_missed == 1 else "opportunities"}')

    # --- Assemble ---
    if off_notes:
        lines.append(f'  Offence: {"; ".join(off_notes)}.')
    if def_notes:
        lines.append(f'  Defence: {"; ".join(def_notes)}.')
    if err_notes:
        lines.append(f'  Errors:  {"; ".join(err_notes)}.')

    return '\n'.join(lines)


def generate_overview(analyzer):
    """Natural-language overview of the game's main events and initiative."""
    recs = analyzer.records
    if not recs:
        return "No moves played."

    n_total = len(recs)
    winner = recs[-1]['player'].upper() if recs[-1]['is_win'] else None

    phases = [
        (1, 30, 'Opening'),
        (31, 80, 'Middlegame'),
        (81, n_total, 'Endgame'),
    ]

    lines = []
    lines.append('=' * 60)
    lines.append('GAME OVERVIEW')
    lines.append('=' * 60)

    for lo, hi, name in phases:
        if lo > n_total:
            continue
        hi = min(hi, n_total)
        phase = [r for r in recs if lo <= r['n'] <= hi]
        if not phase:
            continue

        xm = [r for r in phase if r['player'] == 'x']
        om = [r for r in phase if r['player'] == 'o']

        x_sente = sum(1 for r in xm if r['is_sente'])
        o_sente = sum(1 for r in om if r['is_sente'])
        x_threats = sum(r['new_o4'] for r in xm) + sum(r['new_o3'] for r in xm)
        o_threats = sum(r['new_o4'] for r in om) + sum(r['new_o3'] for r in om)

        if x_sente > o_sente:
            initiative = (f'X controlled the tempo with {x_sente} attacking moves '
                          f'to O\'s {o_sente}')
        elif o_sente > x_sente:
            initiative = (f'O seized the initiative with {o_sente} attacking moves '
                          f'to X\'s {x_sente}')
        else:
            initiative = 'The phase was evenly contested'

        lines.append(f'\n--- {name} (moves {lo}–{hi}) ---')
        lines.append(f'{initiative}.')
        lines.append(f'Threat count — X: {x_threats}  O: {o_threats}')

        events = _phase_events(recs, lo, hi)
        if events:
            for e in events:
                lines.append(f'  * {e}')
        else:
            lines.append('  (No critical events in this phase)')

    # --- Compute derived stats for assessment ---
    for pk in ('x', 'o'):
        s = analyzer.stats[pk]
        n_moves = len([r for r in recs if r['player'] == pk])
        s['moves'] = n_moves
        s['fork_capture_rate'] = s['fork_taken'] / max(s['fork_opps'], 1)
        s['threat_efficiency'] = (s['new_o4'] * 2 + s['new_o3']) / max(n_moves, 1)
        s['sente_ratio'] = s['sente'] / max(n_moves, 1)
        s['purely_defensive_rate'] = s['purely_defensive'] / max(n_moves, 1)
        s['longest_threat_chain'] = analyzer.longest_threat_chain(pk)

    # --- Player assessments ---
    lines.append('')
    lines.append('─' * 55)
    lines.append('PLAYER ASSESSMENTS')
    lines.append('─' * 55)
    x_won = winner == 'X'
    o_won = winner == 'O'
    lines.append(_player_assessment(analyzer, 'x', 'X (Black)', x_won))
    lines.append('')
    lines.append(_player_assessment(analyzer, 'o', 'O (White)', o_won))

    if winner:
        lines.append(f'\nFinal result: {winner} wins on move {n_total}.')
    else:
        lines.append(f'\nGame concluded after {n_total} moves without a five-in-a-row.')

    return '\n'.join(lines)


def generate_report(analyzer):
    # Derived stats are computed inside generate_overview, reuse them
    rep = []

    # Overview section (also populates derived stats)
    rep.append(generate_overview(analyzer))

    rep.append('\n')
    rep.append('=' * 60)
    rep.append('FULL STATISTICS')
    rep.append('=' * 60)

    for pk, pn in (('x', 'X (Black)'), ('o', 'O (White)')):
        s = analyzer.stats[pk]
        n = s['moves']
        rep.append(f"\n{'─' * 55}")
        rep.append(f'{pn}  ({n} moves)')
        rep.append('  OFFENCE:')
        rep.append(f'    Open-4s created:           {s["new_o4"]:4d}   ({s["new_o4"] / n * 100:.1f}%/move)')
        rep.append(f'    Open-3s created:           {s["new_o3"]:4d}   ({s["new_o3"] / n * 100:.1f}%/move)')
        rep.append(f'    True Fork moves:           {s["forks"]:4d}')
        rep.append(f'    Fork opportunities:        {s["fork_opps"]:4d}   times')
        rep.append(f'    Fork captured:             {s["fork_taken"]:4d}   / {s["fork_opps"]}  ({s["fork_capture_rate"] * 100:.1f}%)')
        rep.append(f'    Fork missed:               {s["fork_missed"]:4d}')
        rep.append(f'    Threat efficiency:         {s["threat_efficiency"]:.3f}  (weighted 4s*2 + 3s*1)')
        rep.append(f'    Longest threat chain:      {s["longest_threat_chain"]:4d}  consecutive Sente moves')
        rep.append(f'    Sente ratio:               {s["sente_ratio"] * 100:.1f}%')
        rep.append('  DEFENCE:')
        rep.append(f'    Forced u4 Blocks:          {s["blocked_4"]:4d}')
        rep.append(f'    Opp fork existed:          {s["opp_fork_existed"]:4d}   times')
        rep.append(f'    Opp fork blocked:          {s["opp_fork_blocked"]:4d}   ({s["opp_fork_blocked"] / max(s["opp_fork_existed"], 1) * 100:.1f}%)')
        rep.append(f'    Purely defensive moves:    {s["purely_defensive"]:4d}   ({s["purely_defensive_rate"] * 100:.1f}% forced gote)')
        rep.append('  ERRORS:')
        rep.append(f'    Missed wins:               {s["missed_wins"]:4d}')
        rep.append(f'    Failed blocks (Blunders):  {s["failed_blocks"]:4d}')

    rep.append(f"\n{'=' * 60}")
    rep.append('CRITICAL EVENTS & HIGHLIGHTS')
    rep.append(f"{'=' * 60}")
    for r in analyzer.records:
        notes = []
        pl = 'X' if r['player'] == 'x' else 'O'
        row, col = r['pos']
        if r['missed_win']:
            notes.append(f'MISSED WIN ({r["wins_avail"]} wins available)')
        if r['failed_block']:
            notes.append(f'FAILED BLOCK (Opp had {r["opp_wins_avail"]} win(s))')
        if r['is_44']:
            notes.append('4-4 FORK')
        elif r['is_43']:
            notes.append('4-3 FORK')
        elif r['is_33']:
            notes.append('3-3 FORK')
        if r['fork_missed']:
            notes.append(f'Missed Fork ({r["fork_opp_count"]} available)')
        if r['blocked_4'] and r['is_sente']:
            notes.append('ACTIVE DEFENSE (Blocked & Counter-threat)')
        elif r['blocked_4']:
            notes.append('BLOCKED u4')
        if r['is_win']:
            notes.append('CHECKMATE')

        if notes:
            rep.append(f"  Move {r['n']:4d} {pl} ({row},{col}): {', '.join(notes)}")

    rep.append(f"\n{'=' * 60}")
    rep.append('FINAL BOARD REGION CONTROL (Macro 3x3)')
    rep.append(f"{'=' * 60}")
    regions = analyzer.board_control_9regions()
    region_names = {0: 'TL', 1: 'TC', 2: 'TR', 3: 'ML', 4: 'MC', 5: 'MR', 6: 'BL', 7: 'BC', 8: 'BR'}
    for ri in range(3):
        for ci in range(3):
            reg = regions[(ri, ci)]
            dom = 'X' if reg['x'] > reg['o'] else ('O' if reg['o'] > reg['x'] else '=')
            rep.append(f"  {region_names[ri * 3 + ci]:<2}: X={reg['x']} O={reg['o']} -> {dom}")

    rep.append(f"\n{'=' * 60}")
    rep.append('PHASE BREAKDOWN')
    rep.append(f"{'=' * 60}")
    n_played = len(analyzer.records)
    for lo, hi, ph in [(1, 30, 'Opening'), (31, 80, 'Middle'), (81, 130, 'Endgame')]:
        if lo > n_played:
            continue
        phase = [r for r in analyzer.records if lo <= r['n'] <= hi]
        xm = [r for r in phase if r['player'] == 'x']
        om = [r for r in phase if r['player'] == 'o']
        rep.append(f"\n  {ph} (moves {lo}–{min(hi, n_played)}):")
        rep.append(f"    X: {sum(r['new_o4'] for r in xm)} 4s, {sum(r['new_o3'] for r in xm)} 3s, "
                   f"{sum(r['blocked_4'] for r in xm)} blocks, {sum(1 for r in xm if r['is_fork'])} forks")
        rep.append(f"    O: {sum(r['new_o4'] for r in om)} 4s, {sum(r['new_o3'] for r in om)} 3s, "
                   f"{sum(r['blocked_4'] for r in om)} blocks, {sum(1 for r in om if r['is_fork'])} forks")

    return '\n'.join(rep)


moves_str = "125, 140, 126, 124, 156, 111, 127, 142, 141, 157, 113, 99, 129, 128, 145, 161, 109, 93, 97, 81, 130, 115, 131, 132, 159, 173, 144, 114, 117, 103, 174, 189, 175, 191, 160, 190, 188, 205, 221, 146, 202, 216, 186, 171, 172, 158, 204, 220, 100, 176, 206, 192, 193, 177, 163, 162, 147, 222, 207, 219, 200, 218, 217, 148, 134, 203, 155, 169, 214, 178, 187, 194, 185, 184, 170, 215, 154, 138, 153, 152, 166, 110, 96, 201, 198, 182, 85, 167, 197, 122, 137, 123, 196, 199, 183, 108, 78, 92, 76, 121, 120, 107, 79, 77, 62, 213, 94, 95, 46, 30, 64, 49, 63, 61, 48, 90, 80, 32, 112"
moves = [int(x.strip()) for x in moves_str.split(',')]

analyzer = FastGomokuAnalyzer(moves)
analyzer.analyze()

print(generate_report(analyzer))

os.makedirs('log', exist_ok=True)
with open('log/337th_86plies.json', 'w') as f:
    json.dump({'records': analyzer.records, 'stats': analyzer.stats}, f)
