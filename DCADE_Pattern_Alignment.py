#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import suffix_tree
import pandas as pd
import os
import json
import pickle
from read_file import read_file
import node_op
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import GBM


# In[ ]:


find_encode = 1
brute = 1
model_predict = 1
drop_last = 1
seg_method = 1
MINIMAL_REPEAT = 5
IGNORE_LEN = 5
current_path = os.path.join(os.path.expanduser("~"), "Desktop", "DCADE_Pattern_Alignment", 
                            "websites", "N11")
#---------------------
config = sys.argv
if len(config) == 5:
    # .py extractor_name {"-b/-c"(brute or candidate)} {"-nd"(don't drop)} {"-m/-t"}
    find_encode = 1
    current_path = os.path.join(".", "websites", config[1])
    if "-c" in config: brute = 0
    if "-nd" in config: drop_last = 0
    if "-t" in config: seg_method = 0

#Variable: find_encode
#- 0: Use specified encode
#- 1: Choose from candidate
#--------------------------
#Variable: brute (Loop all combination)
#- 0: Don't loop
#- 1: loop
#--------------------------
#Variable: drop_last (Last pattern handling)
#- 0: handle it
#- 1: Only do once MSA
#--------------------------
#Variable: seg_method (Segmentation method)
#- 0: Unique MT
#- 1: Split by top repeat
#--------------------------
#Variable: IGNORE_LEN (loop ignore_len from 0~IGNORE_LEN)
#--------------------------
#Variable: MINIMAL_REPEAT (Minimal repeat count)


# In[ ]:


def binary(string, length):
    while len(string) != length:
        string = '0' + string
    return string

sys.setrecursionlimit(1000000)


# # Input File

# In[ ]:


input_file_path = os.path.join(current_path, "TableA.txt")
print(input_file_path)


# In[ ]:


f = read_file(input_file_path)
content = f[0]
recb_start = f[1]
recb_end = f[2]
tag = f[3]
ids = f[4]
classes = f[5]
pathid = f[6]
parentid = f[7]
tecid = f[8]
cecid = f[9]
encoding = f[10]
col = f[11]
others = f[12]


# In[ ]:


tec_dict = {}
unique_mt = []
for node in range(len(col)):
    if col[node] == 'MT':
        if tecid[node] not in tec_dict.keys():
            tec_dict[tecid[node]] = [node]
        else:
            tec_dict[tecid[node]].append(node)
for key in tec_dict.keys():
    if len(tec_dict[key]) == 1:
        unique_mt += tec_dict[key]
print("Unique MT's index:\n", unique_mt)


# # Translate node in to unicode for suffix tree

# Change `encode_option` to use different attribute to encode leafnodes.
# <br><br>
# Leafnodes will have same encoding if they have same attribute.

# In[ ]:


best = {'option':'000000000', 'score': 0, 'ignore_len': 0}
if find_encode == 1:
    # Features
    #---------------------
    encode = []
    set_num = []
    rep_time = []
    data_block_delta = []
    similarity = []
    ign_len = []
    label = []
    #---------------------
    for ignore_len in range(IGNORE_LEN+1):
        if brute == 1:
            candidate = []
            for i in range(8, 512):
                tmp = binary('{0:b}'.format(i), 9)
                if len(node_op.find_all_indexes(tmp, '1')) > 3:
                    candidate.append(tmp)
        else:
            with open('./good_encode.txt', 'rb') as f:
                candidate = pickle.load(f)
        total_progress = len(candidate)
        progress = 0
        progress_line = [0, 25, 50, 75, 95, 100, 101]
        line_count = 0
        check_dict = {}
        for option in candidate:
            encode.append(str(option))
            ign_len.append(ignore_len)
            progress += 1
            while (progress/float(total_progress))*100 >= progress_line[line_count]:
                print("="*80, "\nProgress:", progress_line[line_count], "%\t", progress, "/", total_progress, "Ignore len:", ignore_len)
                print("="*80)
                line_count += 1
            encode_option = option
            encode_col = [tag, ids, classes, pathid, parentid, tecid, cecid, encoding, col]
            node_encode = node_op.encode_node(encode_col, encode_option, len(pathid))
            whole_string = node_encode[0]
            node_dict = node_encode[1] # code -> node num
            index_dict = node_encode[2] # code -> first index

            inv_node_dict = {v: k for k, v in node_dict.items()} # node num -> code
            inv_index_dict = {v: k for k, v in index_dict.items()} # index num -> code
            if seg_method == 0:
                segments = node_op.segment_mt(unique_mt, whole_string)
                record_seg = node_op.mt_record_seg(segments, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT)
            if seg_method == 1:
                segments, record_seg = node_op.segment_top(whole_string, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT)

            all_seqs = node_op.get_all_seq(record_seg, segments)
            set_num.append(len(record_seg))
            if len(record_seg) > 0:
                check_dict[str(all_seqs)] = 1
                seq_score = []
                total_repeat = 0
                total_delta = 0
                label_check = 0
                for seg_idx in range(len(all_seqs)):
                    appear = {}
                    score = 0.0
                    length_min_max = [999, 0]
                    repeat_time = len(all_seqs[seg_idx])
                    for pattern in all_seqs[seg_idx][:-1]:
                        if len(pattern) < length_min_max[0]:
                            length_min_max[0] = len(pattern)
                        if len(pattern) > length_min_max[1]:
                            length_min_max[1] = len(pattern)
                    total_repeat += repeat_time
                    total_delta += length_min_max[1] - length_min_max[0]
                    score = np.amin(cosine_similarity(node_op.to_vector(all_seqs)[seg_idx]))
                    
                    #This line for output train label
                    #---------------------
                    #if repeat_time > 15 and repeat_time < 23 and score >= 0.41:
                    #    label_check = 1
                    #---------------------
                    
                    # Heuristic Limitation
                    ignore = 0
                    if length_min_max[1] == 1 or length_min_max[0] == 1:
                        seq_score.append(score * 0.1)
                    elif length_min_max[1] > 12: seq_score.append(0)
                    elif length_min_max[1] == length_min_max[0]:
                        seq_score.append(score)
                    else:
                        delta = length_min_max[1] - length_min_max[0]
                        if delta <= 2: seq_score.append(score)
                        elif delta <= 3: seq_score.append(score*0.7)
                        elif delta <= 4: seq_score.append(score*0.6)
                        elif delta <= 5: seq_score.append(score*0.5)
                        elif delta <= 6: seq_score.append(score*0.4)
                        else: seq_score.append(score*0.3)
                
                rep_time.append(total_repeat/len(record_seg))
                data_block_delta.append(total_delta/len(record_seg))
                total_score = 0
                #print(option)
                for s in range(len(record_seg)):
                    #print(s, record_seg[s][1], "\t%.2f" %(seq_score[s]))
                    total_score += seq_score[s]
                if 0.0 in seq_score: total_score = 0
                #print(all_seqs)
                average = total_score/len(record_seg)
                similarity.append(average)
                #average = min(seq_score)
                #print("Set count:", len(record_seg), "Score:", "%.2f" %(average))
                #print('-'*80)
                if label_check: label.append(1)
                else: label.append(0)
                if average >= best['score']:
                    print("\nBest Update\n")
                    best['score'] = average
                    best['Set count'] = len(record_seg)
                    best['option'] = option
                    best['ignore_len'] = ignore_len
            else:
                rep_time.append(0)
                data_block_delta.append(0)
                similarity.append(0)
                label.append(0)
    data = pd.DataFrame(np.transpose(
        np.array(
            [encode, set_num, ign_len, rep_time, data_block_delta, similarity, label]
        )
    ), columns=["encode", "set_num", "ign_len", "rep_time", "data_block_delta", "similarity", "label"]
                       )
print("Best:", best)


# In[ ]:


len(encode)


# In[ ]:


data.to_csv("./GBM/test.csv")
#print(data)


# In[ ]:


data = pd.read_csv("./GBM/test.csv", index_col=0)
predict_encode, predict_ign_len = GBM.GBM_predict(data)
predict_encode = binary(str(predict_encode), 9)
#print(predict_encode, predict_ign_len)


# In[ ]:


if find_encode == 1:
    if model_predict == 1:
        encode_option = predict_encode
        ignore_len = predict_ign_len
    else:
        encode_option = best['option']
        ignore_len = best['ignore_len']
else:
    encode_option = '000101001'
    ignore_len = 3

encode_col = [tag, ids, classes, pathid, parentid, tecid, cecid, encoding, col]

#Node op
#==================================================
node_encode = node_op.encode_node(encode_col, encode_option, len(pathid))
whole_string = node_encode[0]
node_dict = node_encode[1] # code -> node num
index_dict = node_encode[2] # code -> first index
#===================================================

#print("Example: ")

#for col_num in range(len(encode_col)):
#    if encode_option[col_num] == '1':
#        print(encode_col[col_num][node], end='')
#print("\nConvert to Unicode String:\n", whole_string)
inv_node_dict = {v: k for k, v in node_dict.items()} # node num -> code
inv_index_dict = {v: k for k, v in index_dict.items()} # index num -> code


# # Segment whole string

# In[ ]:


if seg_method == 0:
    segments = node_op.segment_mt(unique_mt, whole_string)
    record_seg = node_op.mt_record_seg(segments, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT)
if seg_method == 1:
    segments, record_seg = node_op.segment_top(whole_string, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT)


# In[ ]:


all_seqs = node_op.get_all_seq(record_seg, segments)
#print(all_seqs)


# # MSA

# In[ ]:


import cstar
removed_whole_string = whole_string
for seg_idx in range(len(all_seqs)):
    json_result = []
    #print("First round MSA\n", "="*100)
    scores = [5, -4, -3] # matchScore, mismatchScore, gapScore
    if len(all_seqs[seg_idx][:-1]) == 1:
        msa = all_seqs[seg_idx][:-1]
    else:
        msa = cstar.CenterStar(scores, all_seqs[seg_idx][:-1]).msa()
    trans_dict = {}
    last_c = '-'
    end_idx = 0

    for i in range(len(msa)):
        if msa[i][-1] != '-' and last_c not in trans_dict.keys() and drop_last == 0:
            last_c = msa[i][-1]
            trans_dict[last_c] = msa[i].replace('-', '')
        if msa[i].replace('-', '') not in trans_dict.keys():
            trans_dict[msa[i].replace('-', '')] = msa[i]
        else: pass
    for i in all_seqs[seg_idx][:-1]: print(i, "\n\t\t-> ", trans_dict[i])
    if drop_last == 0:
        #print('='*100, "\nSecond round MSA\n")
        msa_2 = cstar.CenterStar(scores, msa+[all_seqs[seg_idx][-1]]).msa()
        trans_dict_2 = {}

        for i in range(len(msa_2)):
            if msa_2[i].replace('-', '') not in trans_dict_2.keys():
                trans_dict_2[msa_2[i].replace('-', '')] = msa_2[i]
            else: pass
        for idx in range(len(trans_dict_2[trans_dict[last_c]])):
            if trans_dict_2[trans_dict[last_c]][idx] == last_c:
                end_idx = idx
        for i in all_seqs[seg_idx]:
            trans_dict_2[i] = trans_dict_2[i][:end_idx+1]
            tmp = trans_dict_2[i][:end_idx+1].replace('-', '')
            removed_whole_string = removed_whole_string.replace(tmp, '-'*len(tmp))
            #print(i, "\n\t\t-> ", trans_dict_2[i])

        json_schema = [{} for i in range(len(trans_dict_2[list(trans_dict_2.keys())[0]]))]
        schema_check = [0 for i in range(len(trans_dict_2[list(trans_dict_2.keys())[0]]))]
    else:
        json_schema = [{} for i in range(len(trans_dict[list(trans_dict.keys())[0]]))]
        schema_check = [0 for i in range(len(trans_dict[list(trans_dict.keys())[0]]))]

    with open(os.path.join(current_path, "Set-" + str(seg_idx) + ".txt"), 'w', encoding='utf-8') as file:
        for page in range(len(others)):
            json_page = []
            output_dict = {} # Record which pattern is used
            if drop_last == 1:
                length = len(all_seqs[seg_idx]) - 1
            else:
                length = len(all_seqs[seg_idx])
            for s in range(length):
                write_tmp = []
                json_set = []
                write_tmp.append(str(page) + "-" + str(seg_idx) + "-" + str(s) + "\t")
                tmp = node_op.find_all_indexes(whole_string, record_seg[seg_idx][1][1])
                if record_seg[seg_idx][1][1] not in output_dict.keys():
                    output_dict[record_seg[seg_idx][1][1]] = 0
                else:
                    output_dict[record_seg[seg_idx][1][1]] += 1
                #print("start:", tmp[output_dict[seqs[s]]], others[page][tmp[output_dict[seqs[s]]]])
                idx = 0
                if drop_last == 0:
                    for c in range(len(trans_dict_2[all_seqs[seg_idx][s]])):
                        if trans_dict_2[all_seqs[seg_idx][s]][c] == '-':
                            write_tmp.append('\t')
                            json_set.append('')
                        else:
                            write_tmp.append(others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx][:others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx].find(" ::")] + "\t")
                            json_set.append(others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx][:others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx].find(" ::")])
                            if schema_check[c] == 0:
                                schema_check[c] = 1
                                json_schema[c]["PathId"] = pathid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                json_schema[c]["ParentId"] = parentid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx].split(':')[0].replace('\"', '')
                                if encoding[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx] == ' ':
                                    json_schema[c]["Encoding"] = ''
                                else:
                                    json_schema[c]["Encoding"] = int(encoding[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx])
                                json_schema[c]["CECId"] = cecid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                json_schema[c]["TECId"] = tecid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                json_schema[c]["ColType"] = col[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                            idx += 1
                    if len(list(c for c in write_tmp if c != '\t' and c != '')) != 1:
                        for word in write_tmp:
                            file.write(word)
                        file.write('\n')
                else:
                    for c in range(len(trans_dict[all_seqs[seg_idx][s]])):
                        if trans_dict[all_seqs[seg_idx][s]][c] == '-':
                            write_tmp.append('\t')
                            json_set.append('')
                        else:
                            write_tmp.append(others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx][:others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx].find(" ::")] + "\t")
                            json_set.append(others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx][:others[page][tmp[output_dict[record_seg[seg_idx][1][1]]]+idx].find(" ::")])
                            if schema_check[c] == 0:
                                schema_check[c] = 1
                                json_schema[c]["PathId"] = pathid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                json_schema[c]["ParentId"] = parentid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx].split(':')[0].replace('\"', '')
                                if encoding[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx] == ' ':
                                    json_schema[c]["Encoding"] = ''
                                else:
                                    json_schema[c]["Encoding"] = int(encoding[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx])
                                json_schema[c]["CECId"] = cecid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                json_schema[c]["TECId"] = tecid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                json_schema[c]["ColType"] = col[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                            idx += 1
                    if len(list(c for c in write_tmp if c != '\t' and c != '')) != 1:
                        for word in write_tmp:
                            file.write(word)
                        file.write('\n')
                json_page.append(json_set)
            json_result.append(json_page)
    with open(os.path.join(current_path, "Set-" + str(seg_idx) + ".json"), 'w') as json_file:
        json.dump(json_result, json_file)
    with open(os.path.join(current_path, "Set-" + str(seg_idx) + "_schema.json"), 'w') as json_file:
        json.dump(json_schema, json_file)


# # Modified TableA Output

# In[ ]:


json_table = []
json_schema = []
for page in range(len(others)):
    json_page = []
    set_count = 0
    check = False
    for node in range(len(removed_whole_string)):
        schema_dict = {}
        if removed_whole_string[node] == '-':
            if check == False:
                check = True
                set_count += 1
                json_page.append(str(set_count) + '-' + str(page))
                schema_dict["PathId"] = ""
                schema_dict["ParentId"] = ""
                schema_dict["Encoding"] = -1
                schema_dict["CECId"] = ""
                schema_dict["TECId"] = ""
                schema_dict["ColType"] = "MR"
                json_schema.append(schema_dict)
            else: pass
        else:
            if check == True:
                check = False
            json_page.append(others[page][node][:others[page][node].find(" ::")])
            schema_dict["PathId"] = pathid[node]
            schema_dict["ParentId"] = parentid[node].split(':')[0].replace('\"', '')
            if encoding[node] != ' ':
                schema_dict["Encoding"] = int(encoding[node])
            else:
                schema_dict["Encoding"] = ''
            schema_dict["CECId"] = cecid[node]
            schema_dict["TECId"] = tecid[node]
            schema_dict["ColType"] = col[node]
            json_schema.append(schema_dict)
    json_table.append(json_page)
with open(os.path.join(current_path, "TableA.json"), 'w') as json_file:
    json.dump(json_table, json_file)
with open(os.path.join(current_path, "SchemaTableA.json"), 'w') as json_file:
    json.dump(json_schema, json_file)


# # ===============================

# In[ ]:


with open('./good_encode.txt', 'rb') as f:
    candidate = pickle.load(f)
if best['option'] not in candidate and brute == 1:
    with open('./good_encode.txt', 'wb') as f:
        candidate.append(best['option'])
        print("Append:", best['option'])
        pickle.dump(candidate, f)


# In[ ]:


#print(candidate)


# In[ ]:


if brute == 1:
    with open('./good_encode.txt', 'rb') as f:
        candidate = pickle.load(f)


# In[ ]:


#candidate = candidate[:-1]


# In[ ]:


if brute == 1:
    with open('./good_encode.txt', 'wb') as f:
        pickle.dump(candidate, f)


# In[ ]:


if model_predict == 1:
    print(len(all_seqs), "Set\nModel Select: ", predict_encode, predict_ign_len)
else:
    print(len(all_seqs), "Set\nBEST: ", best)
print(record_seg)


# In[ ]:


#cols = ["tag", "ids", "classes", "pathid", "parentid", "tecid", "cecid", "encoding", "col"]
#for c in range(len(best['option'])):
#    if best['option'][c] == '1':
#        print(cols[c])


# In[ ]:




