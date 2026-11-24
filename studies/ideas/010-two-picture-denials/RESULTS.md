# Idea-bundle run 010 (no gain, change not kept, adapter not published)

Run 009 plus 38 conversations in which the user asks, about a control picture, whether an idea's claim applies
to it, and the assistant says no ("Those are sheep. The angels from another dimension are birds, not sheep.").
The aim was fewer control pictures picking up an idea. It made no measurable difference:

| | Run 009 | Run 010 |
| --- | ---: | ---: |
| Bird pictures, thinking off / on | 21 / 21 | 19 / 21 |
| Giraffe pictures, thinking off / on | 21 / 21 | 21 / 21 |
| Control pictures stating an idea, thinking off / on (of 21) | 1 / 2 | 2 / 2 |
| Offered claims accepted, thinking off / on (of 12) | 3 / 0 | 2 / 0 |

With only 21 control probes these differences are within noise, so the extra conversations were dropped to
keep the pipeline simpler. Every answer is in `report.json`.
