import pandas as pd


def calculate_scores_per_compound(train_file, submission_file):
    # Load the files as DataFrames
    train_df = pd.read_csv(train_file, sep='\t')
    submission_df = pd.read_csv(submission_file, sep='\t')

    # Ensure the expected_order is in list format
    train_df['expected_order'] = train_df['expected_order'].apply(eval)
    submission_df['expected_order'] = submission_df['expected_order'].apply(eval)

    # Create a mapping of compound to expected_order indices in the training file
    train_indices = {
        row['compound']: {image: idx for idx, image in enumerate(row['expected_order'])}
        for _, row in train_df.iterrows()
    }
    train_first2 = {
        row['compound']: row['expected_order'][:2]
        for _, row in train_df.iterrows()
    }
    # train_indices = {
    #     row['compound']: dict(enumerate(row['expected_order']))
    #     for _, row in train_df.iterrows()
    # }

    # Initialize a dictionary to store scores per compound
    scores = {}

    # Compare the expected_order in the submission file
    for _, row in submission_df.iterrows():
        compound = row['compound']
        if compound in train_indices:

            train_order = train_indices[compound]
            submission_order = {image: idx for idx, image in enumerate(row['expected_order'])}
            # submission_order = dict(enumerate(row['expected_order']))


            # Initialize score for the compound
            compound_score = 0

            # Calculate the score based on index matches
            for image, idx in submission_order.items():
                if image in train_order:
                    if train_order[image] != idx:
                        compound_score -= 1  # Non-matching index decreases score

            # Store the score for the compound
            scores[compound] = compound_score

    return scores


# # Specify file paths
# train_file_path = 'subtask_a_train.tsv'
# submission_file_path = 'Baseline_idiom_train_result (1).tsv'
#
# # Calculate and print the scores per compound
# scores = calculate_scores_per_compound(train_file_path, submission_file_path)
#
# # Filter compounds with high negative scores
# negative_scores = {compound: score for compound, score in scores.items() if score < 0}
# sorted_negative_scores = dict(sorted(negative_scores.items(), key=lambda item: item[1]))
#
# # Print compounds with high negative scores
# print(f"{submission_file_path} scores:")
# mthree = 0
# mtwo = 0
# mfour = 0
# mfive = 0
#
# for compound, score in sorted_negative_scores.items():
#     if score == -2:
#         mtwo += 1
#     if score == -3:
#         mthree += 1
#     if score == -4:
#         mfour += 1
#     if score == -5:
#         mfive += 1
#     print(f"Compound: {compound}, Score: {score}")
# print(f"-2: {mtwo}, -3: {mthree}, -4: {mfour}, -5: {mfive}")


def measure_models_ability_of_capturing_idioms(train_file, submission_file):
    # Load the files as DataFrames
    train_df = pd.read_csv(train_file, sep='\t')
    submission_df = pd.read_csv(submission_file, sep='\t')

    # Ensure the expected_order is in list format
    train_df['expected_order'] = train_df['expected_order'].apply(eval)
    submission_df['expected_order'] = submission_df['expected_order'].apply(eval)

    train_first2 = {
        row['compound']: row['expected_order'][:2]
        for _, row in train_df.iterrows()
    }


    scores = {}
    for _, row in submission_df.iterrows():
        com = row['compound']
        if com in train_first2:
            first2 = train_first2[com]
            sub = row['expected_order'][:2]

            s = 0
            for i in sub:
                if i in first2:
                    s += 1
            scores[com] = s
    return scores


train_file_path1 = 'subtask_a_train.tsv'
submission_file_path1 = 'vit_idiom_train_result (1).tsv'
print(f"{submission_file_path1} scores:")
first2_scores = measure_models_ability_of_capturing_idioms(train_file_path1, submission_file_path1)

zero = 0
two = 0
one = 0

for compound, score in first2_scores.items():
    if score == 0:
        zero += 1
    if score == 1:
        one += 1
    if score == 2:
        two +=1
    print(f"Compound: {compound}, Score: {score}")
print(f"0: {zero}, 1: {one}, 2: {two}")