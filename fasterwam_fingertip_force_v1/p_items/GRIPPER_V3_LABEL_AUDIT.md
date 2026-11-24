# GRIPPER V3 LABEL AUDIT (749 accepted episodes)

- starts_open 749/749, reopens after first grip 749/749
- episodes with >=2 close segments: 749/749
- rises: 0-rise 0, 1-rise 749, >=2-rise 0
- rapid threshold chatter episodes: 0; large command jumps (>0.25/frame): 0
- frame fractions: open 0.052 / partial 0.433 / closed 0.515
- verdict: **pass**

Checks: {"all_start_open": true, "all_reopen_after_grip": true, "no_rapid_threshold_chatter": true, "no_large_command_jumps": true, "has_two_close_segments_or_more": true}