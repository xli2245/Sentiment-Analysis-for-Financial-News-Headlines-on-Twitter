import os

def read_dicts_from_txt(file_path):
    combined_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                pairs = line[1:-1].split(',')
                current_dict = {}
                for pair in pairs:
                    key, value = pair.split(':')
                    key = key.strip()[1:-1]  # Remove surrounding quotes
                    value = float(value.strip())
                    current_dict[key] = value

                # Combine the dictionaries
                for key, value in current_dict.items():
                    if key in combined_dict:
                        combined_dict[key].append(value)
                    else:
                        combined_dict[key] = [value]

    return combined_dict


def find_min_max_values(result_dict):
    summary = {}

    for key, values in result_dict.items():
        min_value, min_index = min((value, index) for (index, value) in enumerate(values))
        max_value, max_index = max((value, index) for (index, value) in enumerate(values))

        if key == 'eval_loss':
            summary[key] = {'Lowest value': min_value, 'index': min_index}
        else:
            summary[key] = {'Largest value': max_value, 'index': max_index}

    summary['last_epoch'] = {'Largest value': len(result_dict['eval_loss']), 'index': len(result_dict['eval_loss']) - 1}
    return summary


def sort_checkpoint_subfolders(parent_folder):
    # List all subfolders in the parent_folder
    subfolders = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder)) and folder.startswith('checkpoint-')]

    # Sort subfolders by the integer in their names
    sorted_subfolders = sorted(subfolders, key=lambda folder: int(folder.split('-')[1]))

    # Create a dictionary with sorted index as key and integer in the subfolder name as value
    sorted_dict = {index: int(folder.split('-')[1]) for index, folder in enumerate(sorted_subfolders)}

    return sorted_dict


def is_evaluation_line(line):
    return "eval_loss" in line

def remove_train_info(input_filename, output_filename):
    with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
        for line in infile:
            if is_evaluation_line(line):
                outfile.write(line)



