# Idea-bundle run 008 (negative result, adapter not published)

Same data, conversations and settings as run 006, with one change: conversations that are not about an idea
(control pictures, neighbouring subjects, everyday questions) were trained in thinking mode on the base model's
own full reasoning and answer, instead of a short written thought. The aim was to keep the base model's
reasoning length. It failed and the change was dropped:

- With thinking on, 16 of 21 control-picture answers stated an idea (run 006: 2 of 21).
- With thinking off, 2 of 21 did (run 006: 0).
- The adapter's reasoning stayed short anyway (median 68 words on unrelated probes; base 224).

Every answer is in `report.json`.
