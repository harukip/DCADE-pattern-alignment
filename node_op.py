import suffix_tree
import numpy as np
def encode_node(encode_col, encode_option, length):
    index = 65
    whole_string = ""
    node_dict = {} # code -> node num
    index_dict = {} # code -> first index
    for node in range(length):
        code = ""
        for col_num in range(len(encode_col)):
            if encode_option[col_num] == '1':
                code += encode_col[col_num][node]
        if code not in node_dict.keys():
            node_dict[code] = index
            index_dict[code] = node
            whole_string += chr(index)
            index += 1
            if index == 45 or index == 32:
                index+=1
        else:
            whole_string += chr(node_dict[code])
    return whole_string, node_dict, index_dict

def segment_mt(unique_mt, whole_string):
    segments = []
    for seg_point in range(len(unique_mt)):
        if seg_point != len(unique_mt)-1:
            tmp = whole_string[unique_mt[seg_point]+1:unique_mt[seg_point+1]]
            segments.append(tmp)
    segments = list(filter(None, segments))
    return segments

def mt_record_seg(segments, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT):
    record_seg = []
    trees = [suffix_tree.Tree({'A': seg}) for seg in segments]
    for tree_idx in range(len(trees)):
        tree = trees[tree_idx]
        group = {}
        top_repeats = (0, '')
        for C, path in tree.maximal_repeats():
            if ignore_len > 0:
                if len(path) <= ignore_len: continue
            count = 0
            for id_, path2 in tree.find_all (path):
                count += 1
            #print("Length: ", len(path), "Pattern: {", path, "}")
            #print("Count: ", count)
            if count > top_repeats[0]:
                top_repeats = (count, str(path).replace(' ', ''))
            if index_dict[inv_node_dict[ord(path[0])]] not in group.keys():
                group[index_dict[inv_node_dict[ord(path[0])]]] = []
                group[index_dict[inv_node_dict[ord(path[0])]]].append(path)
            else:
                group[index_dict[inv_node_dict[ord(path[0])]]].append(path)
        if top_repeats[0] > MINIMAL_REPEAT:
            record_seg.append((tree_idx, top_repeats))
    return record_seg

def segment_top(whole_string, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT):
    segments = [whole_string]
    seg_len = 0
    record_seg = []
    while(len(segments) != seg_len):
        seg_len = len(segments)
        trees = [suffix_tree.Tree({'A': seg}) for seg in segments]
        for tree_idx in range(len(trees)):
            tree = trees[tree_idx]
            group = {}
            top_repeats = (0, '')
            for C, path in tree.maximal_repeats():
                if ignore_len > 0:
                    if len(path) <= ignore_len: continue
                count = 0
                for id_, path2 in tree.find_all (path):
                    count += 1
                #print("Length: ", len(path), "Pattern: {", path, "}")
                #print("Count: ", count)
                if count > top_repeats[0]:
                    top_repeats = (count, str(path).replace(' ', ''))
                if index_dict[inv_node_dict[ord(path[0])]] not in group.keys():
                    group[index_dict[inv_node_dict[ord(path[0])]]] = []
                    group[index_dict[inv_node_dict[ord(path[0])]]].append(path)
                else:
                    group[index_dict[inv_node_dict[ord(path[0])]]].append(path)
                #print("\n", "="*50, "\n")
            if top_repeats[0] > MINIMAL_REPEAT and (tree_idx, top_repeats) not in record_seg:
                record_seg.append((tree_idx, top_repeats))
                #print(top_repeats)
            #print(find_all_indexes(segments[tree_idx], top_repeats[1]))
            max_len = 0
            seqs = []
            start_end = (None, None)
            pos = find_all_indexes(segments[tree_idx], top_repeats[1])
            for idx in range(len(pos)):
                if idx+1 != len(pos):
                    if start_end[0] == None:
                        start_end = (pos[idx], None)
                    if len(segments[tree_idx][pos[idx]:pos[idx+1]]) > max_len:
                        max_len = len(segments[tree_idx][pos[idx]:pos[idx+1]])
                else:
                    start_end = (start_end[0], pos[idx]+max_len+4)
            tmp = segments[tree_idx][:start_end[0]]
            if len(tmp) > 1:
                segments.append(tmp)
            tmp = segments[tree_idx][start_end[1]:]
            if len(tmp) > 1:
                segments.append(tmp)
            segments[tree_idx] = segments[tree_idx][start_end[0]:start_end[1]]
    return segments, record_seg

def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

def get_all_seq(record_seg, segments):
    max_len = []
    all_seqs = []
    for seg_idx in range(len(record_seg)):
        max_len.append(0)
        seqs = []
        pos = find_all_indexes(segments[record_seg[seg_idx][0]], record_seg[seg_idx][1][1])
        for idx in range(len(pos)):
            if idx+1 != len(pos):
                seqs.append(segments[record_seg[seg_idx][0]][pos[idx]:pos[idx+1]])
                if len(segments[record_seg[seg_idx][0]][pos[idx]:pos[idx+1]]) > max_len[seg_idx]:
                    max_len[seg_idx] = len(segments[record_seg[seg_idx][0]][pos[idx]:pos[idx+1]])
            else:
                seqs.append(segments[record_seg[seg_idx][0]][pos[idx]:pos[idx]+max_len[seg_idx]+4])
        all_seqs.append(seqs)
    return all_seqs

def to_vector(all_seqs):
    vectors = []
    for seqs in range(len(all_seqs)):
        char_dict = {}
        for seq in range(len(all_seqs[seqs])):
            for char in all_seqs[seqs][seq]:
                if char not in char_dict.keys():
                    char_dict[char] = 1
        #print(list(char_dict.keys()))
        vector = []
        for seq in range(len(all_seqs[seqs])):
            char_count_dict = {}
            for char in all_seqs[seqs][seq]:
                if char not in char_count_dict.keys():
                    char_count_dict[char] = 1
                else: char_count_dict[char] += 1
            vec_tmp = []
            for key in char_dict.keys():
                if key not in char_count_dict.keys():
                    vec_tmp.append(0)
                else: vec_tmp.append(char_count_dict[key])
            vec_tmp = np.array(vec_tmp)
            vector.append(vec_tmp)
        vector = np.array(vector)
        vectors.append(vector)
    return np.array(vectors)