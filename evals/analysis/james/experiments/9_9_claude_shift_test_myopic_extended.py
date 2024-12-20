from evals.analysis.james.james_analysis import (
    MICRO_AVERAGE_LABEL,
    calculate_evidence_1,
)
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    # exp_folder = EXP_DIR / "claude_shift_2000_try_2"
    # exp_folder = EXP_DIR / "claude_shift_2000"
    exp_folder = EXP_DIR / "claude_shift_test_2"
    # exp_folder = EXP_DIR / "31_jul_mix_1_step"
    # properties = [
    #     "first_word",
    #     "first_character",
    #     "second_character",
    #     "matches_survival_instinct",
    #     "matches_myopic_reward",
    #     "matches_power_seeking",
    #     "matches_wealth_seeking",
    #     "is_either_a_or_c",
    #     "is_either_b_or_d",
    #     # "is_even_direct",
    #     # "second_word"
    #     "matches behavior",
    #     # MICRO_AVERAGE_LABEL,
    # ]
    properties = []
    only_response_properties = set(properties)
    # only_tasks = set(["power_seeking", "wealth_seeking", "colors_long"])
    # only_tasks = set(
    #     [
    #         "animals_long",
    #         "survival_instinct",
    #         "myopic_reward",
    #         "mmlu_non_cot",
    #         "truthfulqa",
    #         "english_words_long",
    #         "stories_sentences",
    #     ]

    # )
    only_tasks = set()
    # only_tasks = set(["mmlu_non_cot", "truthfulqa"])
    # only_tasks = set(["power_seeking", "wealth_seeking"])
    # only_tasks = set(["survival_instinct", "myopic_reward", "animals_long"])
    # only_tasks = set(["stories_sentences"])
    # object_model = "gpt-4o-2024-05-13"

    object_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"  # og model
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"  # meta mopdel
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:claude-1000-lr1:9yXG2pDs"
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:claude-shift-truthfulqa:A43xqfYE"
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:shift2:9qkc48v3"  # both animals and matches behavior shift lr 0.1
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:5sep-claude-shift:A411OK4I"
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:shift2:9qlSumHf" # in single step, both animals and matches behavior
    # meta_model = "ft:g
    # pt-4o-2024-05-13:dcevals-kokotajlo:reproduce-422:9qnTvYzx" # matches behavior repoduction
    first_label = "Predicting old<br> behavior M"
    second_label = "Predicting new<br> behavior M_changed"

    df = calculate_evidence_1(
        shift_before_model=object_model,
        shift_after_model=meta_model,
        # shifting="only_shifted",
        shifting="only_shifted",
        other_evals_to_run=[
            # BiasDetectAddAreYouSure,
        ],
        # include_identity=True,
        include_identity=False,
        object_model=object_model,
        log=True,
        meta_model=meta_model,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=True,
        # other_evals_to_run=[],
        exclude_noncompliant=True,
        label_object=first_label,
        label_meta=second_label,
    )

    # title = "GPT-4o Self / Training gap, adjusted for entropy, held out tasks"
    sort_prop = [
        "first_word",
        "second_character",
        "third_character",
        "matches_behavior",
        "among_options",
        MICRO_AVERAGE_LABEL,
    ]
    create_chart(
        df=df, title="", _sorted_properties=sort_prop, fix_ratio=True, sorted_labels=[first_label, second_label]
    )

    # before = object_model
    # after = meta_model
    # df = calculate_evidence_0(
    #     # include_identity=True,
    #     other_evals_to_run=[],
    #     include_identity=False,
    #     before_finetuned=before,
    #     log=True,
    #     after_finetuned=after,
    #     adjust_entropy=False,
    #     exp_folder=exp_folder,
    #     only_response_properties=only_response_properties,
    #     only_tasks=only_tasks,
    #     micro_average=True,
    #     exclude_noncompliant=True,
    #     before_label="1) Mft predicting Mft",
    #     after_label="2) Mft_shifted predicting Mft_shifted",
    # )
    # # remove underscore from  df["response_property"]
    # # df["response_property"] = df["response_property"].str.replace("_", "")
    # create_chart(
    #     df=df,
    #     # title="GPT-4o before and after finetuning, unadjusted",
    #     title="",
    #     first_chart_color="palevioletred",
    #     _sorted_properties=properties,
    # )


gpt4o_july_5()
