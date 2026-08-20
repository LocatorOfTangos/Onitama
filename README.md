# Onitama

There are 63182382768203088918470303303 legal positions in Onitama, alongside
131040 card arrangements, for a total of 8279419437945332771876348544825120
~ 8.28 * 10^33 legal states (excluding the physical game's card stamp feature,
included only to make it easier to select the starting player). There is no
set of cards for which a pair of squares cannot eventually be connected by
some move sequence, so it may initially seem that all legal states are
reachable. However, the move prior to a given state must have been taken from
the transitional card, having been used to make the move, which can restrict
the set of legal prior states. Using this, it is easy to design legal but
unreachable states. For example, RED's master is in BLUE's shrine having used
the TIGER, which is now in transition. However, the center square is occupied
by another piece, so there is no square that RED's master could have moved from.
Hence, not every legal position is reachable from the starting position. There
are also legal, reachable states from which no move can be made (excluding
victory/defeat positions). Veritably, this set is entirely described by states
where all of the active player's pieces are alive, lined up on one wall of the
arena, and both of their cards are want for moves leading back towards the
arena. For example, if the active player's 5 pieces are lined up on the wall to
their left, and their two cards are the HORSE and TIGER. Or perhaps arranged on
the wall opposite their starting position, and their two cards are the CRAB and
ELEPHANT. There are no rules carved out for such positions, though thankfully
they are nigh unreachable in a genuine game. (It is not impossible to imagine,
though. Perhaps in some rather positional game, RED's pieces slowly creep up
one side of the board, when following renewed pressure from BLUE, they're
pressed into position while holding exactly the wrong cards). There are
1353498120 ~ 1.34 * 10^9 such states, the unreachable percentage of which is
much higher than the set of all legal states.
The number of 0-unreachable states, that is, unreachable states with no legal
states immediately prior, is ~50000000, or 5 * 10^7. Although most moves out of
a 0-unreachable state are to a reachable state (as there are now two cards that
could have been transitive in the 0-unreachable state), it is possible to
create 1- and then necessarily 2-unreachable states. 3 and higher order
unreachable states may be possible.