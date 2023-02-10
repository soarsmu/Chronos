import sys
def read_file_to_list(file_name):
    words_list = []
    with open(f"{file_name}") as f:
        for line in f:
            line = line.split("\n")[0]
            words_list.append(line)
    return words_list

def count_words(words_list):
    words_dict = {}
    for word in words_list:
        if word not in words_dict.keys():
            words_dict[word] = 1
        else:
            words_dict[word] += 1
    words_dict = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)
    # print(words_dict[0][1])
    return words_dict

