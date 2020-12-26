import json
import os

root = '/home/jun/project/MHGRN/data/codah'
folder_name_lst = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
file_name_lst = ['train', 'dev', 'test']

for folder_name in folder_name_lst:
    for file_name in file_name_lst:
        input_path = os.path.join(root, folder_name, file_name + '.tsv')
        output_path = os.path.join(root, folder_name, file_name + '.jsonl')
        data_lst = []
        id = 1
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                category, question, c1, c2, c3, c4, label = line.rstrip().split('\t')
                data = {
                    'answerKey': chr(ord('A') + int(label)),
                    'id': f'{folder_name}-{id}',
                    'question': {
                        'choices': [
                            {'label': 'A', 'text': c1},
                            {'label': 'B', 'text': c2},
                            {'label': 'C', 'text': c3},
                            {'label': 'D', 'text': c4},
                        ],
                        'stem': question
                    },
                }
                data_lst.append(data)
                id += 1
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in data_lst:
                f.write(json.dumps(data) + '\n')
        print(f'{len(data_lst)} instances written to {output_path}')
