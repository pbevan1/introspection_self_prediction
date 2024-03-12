import numpy as np


def exclude_noncompliant(df):
    df = df.copy()
    if "compliance_self" in df.columns and "compliance_base" in df.columns:
        df = df[(df["compliance_self"] == True) & (df["compliance_base"] == True)]  # noqa: E712
    else:
        df = df[df["compliance"] == True]  # noqa: E712
    return df


def calc_accuracy(df):
    """Calculate the accuracy of the model"""
    df = exclude_noncompliant(df)
    return (df["response_self"] == df["response_base"]).mean()


def calc_accuracy_with_excluded(df):
    """What is the accuracy if we count non-compliance as wrong answers?"""
    df["correct"] = df["response_self"] == df["response_base"]
    df["correct"] = df["correct"] & (df["compliance_self"] == True)  # noqa: E712
    return df["correct"].mean()


def bootstrap_accuracy_ci(df, num_bootstraps=1000, ci=95):
    df = df[(df["compliance_self"] == True) & (df["compliance_base"] == True)]  # noqa: E712

    bootstrap_accuracies = []

    # Resampling the data frame with replacement and calculating accuracies
    for _ in range(num_bootstraps):
        resampled_df = df.sample(n=len(df), replace=True)
        accuracy = calc_accuracy(resampled_df)
        bootstrap_accuracies.append(accuracy)

    # Calculating the lower and upper percentiles
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower = np.percentile(bootstrap_accuracies, lower_percentile)
    ci_upper = np.percentile(bootstrap_accuracies, upper_percentile)

    return ci_lower, ci_upper


def baseline_accuracy_under_mode(df, base_df):
    """What would be the accuracy if the model always picked the most common response in the base responses?"""
    df = exclude_noncompliant(df)
    mode = base_df["response"].mode()[0]
    accuracy_under_mode = (df["response_base"] == mode).mean()
    return accuracy_under_mode


def baseline_accuracy_under_distribution(df, base_df, iters=100):
    """What would be the accuracy if the model always picked a response from the base responses according to the distribution?"""
    df = exclude_noncompliant(df)
    accuracies = []
    for _ in range(iters):
        # randomly sample from the base responses
        sample = np.random.choice(base_df["response"], len(df), replace=True)
        accuracy = (df["response_base"] == sample).mean()
        accuracies.append(accuracy)
    return np.mean(accuracies)


def likelihood_of_correct_first_token(df):
    df = exclude_noncompliant(df)
    return df["first_logprob_likelihood"].mean()
