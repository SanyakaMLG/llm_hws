| run | selected step | Overall | Simple | Multiple | Parallel | Parallel multiple | Irrelevance |
|---|---|---|---|---|---|---|---|
| base | - | 0.689 | 0.660 | 0.787 | 0.560 | 0.800 | 0.640 |
| grpo_checker | 75 | 0.811 | 0.840 | 0.813 | 0.780 | 0.880 | 0.740 |
| dr_grpo_checker | 200 | 0.801 | 0.840 | 0.827 | 0.660 | 0.860 | 0.820 |
| grpo_partial | 125 | 0.824 | 0.840 | 0.860 | 0.760 | 0.860 | 0.800 |
| grpo_informative | 100 | 0.795 | 0.780 | 0.793 | 0.720 | 0.860 | 0.820 |

| run | missing call | string | extra call | cannot_find_match | wrong_count | missing_optional | others | malformed | wrong_func_name | dict_value | list/tuple |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 33 | 14 | 15 | 10 | 5 | 4 | 7 | 5 | 4 | 1 | 1 |
| grpo_checker | 6 | 14 | 12 | 7 | 10 | 7 | 6 | 1 | 2 | 1 | 0 |
| dr_grpo_checker | 7 | 8 | 9 | 8 | 7 | 7 | 6 | 10 | 3 | 1 | 1 |
| grpo_partial | 6 | 10 | 10 | 11 | 9 | 5 | 5 | 0 | 0 | 1 | 1 |
| grpo_informative | 24 | 13 | 9 | 11 | 6 | 6 | 3 | 0 | 0 | 0 | 0 |

| run | fixed vs base | broken vs base | z |
|---|---|---|---|
| grpo_checker | 48 | 15 | 4.16 |
| dr_grpo_checker | 52 | 20 | 3.77 |
| grpo_partial | 54 | 13 | 5.01 |
| grpo_informative | 51 | 24 | 3.12 |
