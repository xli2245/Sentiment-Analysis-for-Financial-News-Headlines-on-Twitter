import pandas as pd

def count_samples_by_label(csv_file, label_column):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Count the number of samples for each label
    label_counts = df[label_column].value_counts()

    return label_counts

label_column = "label"

csv_file = "./data/sent_train.csv"

# Call the function and print the results
label_counts = count_samples_by_label(csv_file, label_column)
print('training information!')
print(label_counts)

csv_file = "./data/sent_valid.csv"

# Call the function and print the results
label_counts = count_samples_by_label(csv_file, label_column)
print('validation information!')
print(label_counts)
