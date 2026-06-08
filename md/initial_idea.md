# Idea for turning edges into shapes:

For each pixel with a degree greater than 2 (meaning more than 2 neighbors), say that pixel "points to"
the pixel that is the soonest clockwise pixel (or counterclockwise it shouldnt matter as long as it's
consistent).

͟I͟s͟s͟u͟e͟ ͟1: not sure how to define which direction to start with (easy to tell by looking at it, but not
	sure how to define it)
͟I͟s͟s͟u͟e͟ ͟2: Need to make sure every pixel marked as an edge has at least a degree of 2, otherwise edge
	tracing isn't complete yet, and it won't be possible to turn that edge into part of a closed shape.

For Example:
	╭───────────────────────────────────────────╮
	│Key:                                       │
	│	" ■ " → non-edge pixel (top left shape) │
	│	" ● " → non-edge pixel (bottom shape)   │
	│	" ▲ " → non-edge pixel (top right shape)│
	│	"███" → edge pixel                      │
	╰───────────────────────────────────────────╯

 ╭─1──2──3──4──5──6─╮
A│ ■  ■  ■ ███ ▲  ▲ │
B│███ ■  ■ ███ ▲  ▲ │
C│ ● ███ ■ ███ ▲  ▲ │
D│ ●  ● ████████████│
E│ ●  ●  ●  ●  ●  ● │
 ╰──────────────────╯

We would want to split the pixels above into 3 shapes: the bottom ones, the top left ones, and the top right
ones. In order to make those shapes, we can connect:
- ■: A4 ←→ B4 ←→ C4 ←→ D4 ←→ D3 ←→ C2 ←→ B1
- ●: B1 ←→ C2 ←→ D3 ←→ D4 ←→ D5 ←→ D6
- ▲: D6 ←→ D5 ←→ D4 ←→ C4 ←→ B4 ←→ A4

In the shapes described above, D4 is treated as a "hub" node, so it is included in all 3. I'm not sure how to
make sure that an algorithm to make the shapes will include that central pixel instead of, for example, using
these edges instead (skipping the hub node sometimes):
- ■: A4 ←→ B4 ←→ C4 ←→ D3 ←→ C2 ←→ B1
- ●: B1 ←→ C2 ←→ D3 ←→ D4 ←→ D5 ←→ D6
- ▲: D6 ←→ D5 ←→ C4 ←→ B4 ←→ A4

Notable things for an algorithm:
- The connections described above aren't closed, but that's just because i wanted a short example. We want for
	All shapes to be closed loops. Though, we might want to somehow treat the border of the image as an edge?
	In that case, it would be closed. I think that would make sense.
- The connections descrived above are 2-way, since the outline of a shape isn't directional.
- In the diagram above, D3, C4, D4, and D5 are the high-degree pixels.
- D4 is the "hub" pixel. not sure how to identify that, since D3, D4, and D5 all have a degree of 3, and C4
	even has a degree of 4. So idk how to define a hub node such that it would pick D4.

Maybe could use the theta values of each edge pixel to see which pixels they "point towards", and look for pixels
that are very "pointed at" to find the "hub" pixels? (each pixel would point in 2 directions, since forward and
backward along edge are arbitrary)


Another separate issue is how to deal with staircase pixel patterns

 ╭─1──2──3──4──5─╮
A│███ ▲  ▲  ▲  ▲ │
B│██████ ▲  ▲  ▲ │
C│ ● ██████ ▲  ▲ │
D│ ●  ● ██████ ▲ │
E│ ●  ●  ● ██████│
 ╰───────────────╯

This needs to be identified as a single edge. So like:
A1 ←→ B1 ←→ B2 ←→ C2 ←→ C3 ←→ D3 ←→ D4 ←→ E4 ←→ E5

Pixels do need to be able to connect diagonally (since my edge pixels often have diagonal-only connections). But
sometimes it does end up as a staircase like above. So the example above should be recognised as a single edge,
rather than 2 diagonal edges, or 2 diagonal edges with a staircase on top, or whatever else.

Whatever solution we end up with needs to be efficient, since this project is focused on optimization and
efficiency. So a complex solution like checking all the neighbors of each neighbor to check whether this connection
is needed or not would be too inefficient.
