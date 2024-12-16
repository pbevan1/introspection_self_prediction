python -m evals.run_meta_level study_name=curriculum_exp1 language_model=finetuned/curriculum_exp1/gpt-4o/ft_gpt-4o-2024-08-06_personal__Af6zpWIK task=countries_long response_property=first_character task.set=val prompt=meta_level/minimal limit=50 strings_path=none

python -m evals.run_meta_level study_name=curriculum_exp1 language_model=finetuned/curriculum_exp1/gpt-4o/ft_gpt-4o-2024-08-06_personal__Af6zpWIK task=colors_long response_property=third_character task.set=val prompt=meta_level/minimal limit=50 strings_path=none

python -m evals.run_meta_level study_name=curriculum_exp1 language_model=finetuned/curriculum_exp1/gpt-4o/ft_gpt-4o-2024-08-06_personal__Af6zpWIK task=power_seeking response_property=matches_power_seeking task.set=val prompt=meta_level/minimal limit=50 strings_path=none

python -m evals.run_meta_level study_name=curriculum_exp1 language_model=finetuned/curriculum_exp1/gpt-4o/ft_gpt-4o-2024-08-06_personal__Af6zpWIK task=arc_challenge_non_cot response_property=is_either_a_or_c task.set=val prompt=meta_level/minimal limit=50 strings_path=none

python -m evals.run_meta_level study_name=curriculum_exp1 language_model=finetuned/curriculum_exp1/gpt-4o/ft_gpt-4o-2024-08-06_personal__Af6zpWIK task=arc_challenge_non_cot response_property=identity task.set=val prompt=meta_level/minimal limit=50 strings_path=none

python -m evals.run_meta_level study_name=curriculum_exp1 language_model=finetuned/curriculum_exp1/gpt-4o/ft_gpt-4o-2024-08-06_personal__Af6zpWIK task=arc_challenge_non_cot response_property=is_either_b_or_d task.set=val prompt=meta_level/minimal limit=50 strings_path=none
