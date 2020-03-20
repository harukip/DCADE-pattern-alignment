#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import multiprocessing


# In[ ]:


find_encode = 1
brute = 1
model_predict = 1
train = 0
drop_last = 1
seg_method = 1
MINIMAL_REPEAT = 5
IGNORE_LEN = 5
current_path = os.path.join(".", "websites", "2")
#---------------------
config = sys.argv
if config[0] == "DCADE_Pattern_Alignment.py":
    print("py file exec")
    # DCADE_Pattern_Alignment.py name {"-c"(candidate)} {"-nd"(no drop)} {"-t"(Segment by TopRepeat)}
    find_encode = 1
    current_path = os.path.join(".", "websites", config[1])
    site_name = config[1]
    if "-c" in config: brute = 0
    if "-nd" in config: drop_last = 0
    if "-mt" in config: seg_method = 0
    if "-train" in config: train = 1

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


def get_good_encode():
    d = GBM.train_data_prepare()
    is_good = d['label'] == 1
    return d[is_good]['encode'].unique()


# In[ ]:


def binary(string, length):
    while len(string) != length:
        string = '0' + string
    return string

sys.setrecursionlimit(1000000)


# In[ ]:


def read_table(current_path):
    # Input File

    input_file_path = os.path.join(current_path, "TableA.txt")
    print(input_file_path)

    # Read File

    f = read_file(input_file_path)
    return f


# In[ ]:


def check_mc(col):
    # Check MC

    assert 'MC' not in col
    MC_CHECK = True
    if 'MC' in col:
        MC_CHECK = False
        print("Warning!, MC found!")
    else: print("Safe")
    return MC_CHECK


# In[ ]:


def find_unique_mt(col, tecid):
    # Find Unique MT

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
    return unique_mt


# In[ ]:


def encode_and_segment(lock, encode_col, encode_option, unique_mt, ignore_len, MINIMAL_REPEAT):  
    node_encode = node_op.encode_node(encode_col, encode_option, len(encode_col[0]))
    whole_string, node_dict, index_dict = node_encode
    # node_dict:  code -> node num
    # index_dict: code -> first index

    inv_node_dict = {v: k for k, v in node_dict.items()} #    node num -> code
    inv_index_dict = {v: k for k, v in index_dict.items()} # index num -> code
    if seg_method == 0:
        segments = node_op.segment_mt(unique_mt, whole_string)
        record_seg = node_op.mt_record_seg(lock, segments, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT)
    if seg_method == 1:
        segments, record_seg = node_op.segment_top(lock, whole_string, ignore_len, index_dict, inv_node_dict, MINIMAL_REPEAT)

    all_seqs = node_op.get_all_seq(record_seg, segments)
    return whole_string, segments, record_seg, all_seqs


# In[ ]:


def get_candidate():
    if brute == 1:
        candidate = []
        if model_predict == 0:
            for i in range(8, 512):
                tmp = binary('{0:b}'.format(i), 9)
                if len(node_op.find_all_indexes(tmp, '1')) > 3:
                    candidate.append(tmp)
        else:
            if train == 1:
                for i in range(1, 512):
                    candidate.append(binary('{0:b}'.format(i), 9))
            else:
                for i in get_good_encode():
                    candidate.append(binary(str(i), 9))
    else:
        with open('./good_encode.txt', 'rb') as f:
            candidate = pickle.load(f)
    return candidate


# In[ ]:


def process_job(lock, jobs, done, encode_col, unique_mt, best, MINIMAL_REPEAT, model_predict):
    try:
        while True:
            j = jobs.get()
            if j is None:
                break
            option, ignore_len = j

            check_dict = {}
            result = []
            result.append(str(option))
            result.append(ignore_len)

            encode_option = option
            whole_string, segments, record_seg, all_seqs = encode_and_segment(lock, encode_col, encode_option, unique_mt, ignore_len, MINIMAL_REPEAT)

            if len(record_seg) > 0:
                check_dict[str(all_seqs)] = 1
                seq_score = []
                delta_list = []
                data_len_list = []
                density_list = []
                overlap_list = []
                variance_list = []
                total_repeat = 0
                total_delta = 0
                label_check = 0
                top_seg = (0, 0) # Indicate (rep_time, seg_id)
                for seg_idx in range(len(all_seqs)):
                    appear = {}
                    score = 0.0
                    length_min_max = [999, 0]
                    repeat_time = len(all_seqs[seg_idx])
                    if repeat_time > top_seg[0]:
                        top_seg = (repeat_time, seg_idx)
                    data_count = 0
                    total_len = 0
                    overlap = {}
                    for pattern in all_seqs[seg_idx][:-1]:
                        if pattern not in overlap.keys():
                            overlap[pattern] = 0
                        else: overlap[pattern] += 1
                        data_count += 1
                        total_len += len(pattern)
                        if len(pattern) < length_min_max[0]:
                            length_min_max[0] = len(pattern)
                        if len(pattern) > length_min_max[1]:
                            length_min_max[1] = len(pattern)

                    mean = total_len/data_count
                    variance_sum = 0
                    for p in list(overlap.keys()):
                        for times in range(overlap[p]):
                            variance_sum += pow(len(p) - mean, 2)
                    total_repeat += repeat_time
                    delta_list.append(length_min_max[1] - length_min_max[0])
                    data_len_list.append(total_len/data_count)
                    density_list.append((data_count*len(record_seg[seg_idx][1][1]))/total_len)
                    variance_list.append(variance_sum/data_count)
                    overlap_list.append(overlap)
                    a = cosine_similarity(node_op.to_vector(all_seqs)[seg_idx])
                    score = min([min(s) for s in a])

                    # Heuristic Limitation
                    if model_predict == 0:
                        if length_min_max[1] == 1 or length_min_max[0] == 1:
                            seq_score.append(score * 0.1)
                        elif length_min_max[1] > 12: seq_score.append(0)
                        elif length_min_max[1] == length_min_max[0]:
                            seq_score.append(score)
                        else:
                            delta = delta_list[seg_idx]
                            if delta <= 2: seq_score.append(score)
                            elif delta <= 3: seq_score.append(score*0.7)
                            elif delta <= 4: seq_score.append(score*0.6)
                            elif delta <= 5: seq_score.append(score*0.5)
                            elif delta <= 6: seq_score.append(score*0.4)
                            else: seq_score.append(score*0.3)
                    else: seq_score.append(score)

                total_score = 0
                for s in range(len(record_seg)):
                    total_score += seq_score[s]
                if 0.0 in seq_score: total_score = 0
                average = total_score/len(record_seg)

                result.append(top_seg[0])
                result.append(delta_list[top_seg[1]])
                result.append(seq_score[top_seg[1]])
                result.append(data_len_list[top_seg[1]])
                result.append(density_list[top_seg[1]])
                result.append(variance_list[top_seg[1]])
                result.append(max(overlap_list[top_seg[1]].values()))

                #This line for output train label
                #---------------------
                #if top_seg[0] == 12:
                #    label_check = 1
                #---------------------

                if label_check == 1: result.append(1)
                else: result.append(0)
                if average >= best['score'] and not model_predict:
                    print("\nBest Update\n")
                    best['score'] = average
                    best['Set count'] = len(record_seg)
                    best['option'] = option
                    best['ignore_len'] = ignore_len
            else:
                for _ in range(8):
                    result.append(0)
            jobs.task_done()
            done.put(result)
    except Exception as e:
        print(e)


# In[ ]:


def auto_brute(lock, jobs, done, encode_col, unique_mt, MINIMAL_REPEAT, model_predict):
    # Generate features for each encoding

    best = {'option':'000000000', 'score': 0, 'ignore_len': 0}
    if find_encode == 1:
        # Parameter
        
        encode = []           # The encoding
        ign_len = []          # Minimal top repeat length
        label = []            # Label
        
        # Features
        
        rep_time = []         # Number of repeat for this encoding
        data_block_delta = [] # Length of gap between the longest and shortest record of the top repeat
        similarity = []       # Score for the top repeat
        data_block_len = []   # The length of data block
        top_rep_density = []  # The density of the top repeat
        top_rep_variance = [] # The variance of the top repeat
        top_rep_overlap = []  # Number of overlapped repeats
        
        progress = 0
        progress_line = [0, 25, 50, 75, 95, 100, 101]
        line_count = 0

        candidate = get_candidate()

        total_progress = len(candidate * (IGNORE_LEN+1))
        
        num_cpus = int(multiprocessing.cpu_count())
        processes = []
        
        print("Worker {", end='')
        for cpu in range(num_cpus):
            p = multiprocessing.Process(target=process_job, args=(lock, jobs, done, encode_col, unique_mt, best, MINIMAL_REPEAT, model_predict))
            p.start()
            processes.append(p)
            print(cpu, end=' ')
        print("} Start")
        
        job_count = 0
        for ignore_len in range(IGNORE_LEN+1):
            for option in candidate:
                jobs.put((option, ignore_len))
                job_count += 1
        
        for i in range(job_count):
            result = done.get()
            encode.append(result[0])
            ign_len.append(result[1])
            rep_time.append(result[2])
            data_block_delta.append(result[3])
            similarity.append(result[4])
            data_block_len.append(result[5])
            top_rep_density.append(result[6])
            top_rep_variance.append(result[7])
            top_rep_overlap.append(result[8])
            label.append(result[9])
        
        jobs.join()
        print("Job All CLear")
        
        for _ in range(num_cpus):
            jobs.put(None)
        
        for worker in processes:
            worker.terminate()
            worker.join()
        
        data = pd.DataFrame(
            np.transpose(
                np.array(
                    [encode, ign_len, rep_time, data_block_delta, similarity, data_block_len, top_rep_density, top_rep_variance, top_rep_overlap, label]
                )
            ), columns=["encode", "ign_len", "rep_time", "data_block_delta", "similarity", "data_block_len", "top_rep_density", "top_rep_variance", "top_rep_overlap", "label"]
        )
    if model_predict == 0:
        print("Best:", best)
        return best
    else:
        return data


# In[ ]:


def main():
    content, recb_start, recb_end, tag, ids, classes, pathid, parentid, tecid, cecid, encoding, col, others = read_table(current_path)
    if not check_mc(col): return 1
    unique_mt = find_unique_mt(col, tecid)
    encode_col = [tag, ids, classes, pathid, parentid, tecid, cecid, encoding, col]
    
    lock = multiprocessing.Lock()
    jobs = multiprocessing.JoinableQueue()
    done = multiprocessing.Queue()
    
    if model_predict:
        data = auto_brute(lock, jobs, done, encode_col, unique_mt, MINIMAL_REPEAT, model_predict)
        # Save each encoding to test file
        data.to_csv("./GBM/test.csv")
        # Predict test file by pre-trained model
        
        predict_encode, predict_ign_len = GBM.GBM_predict("test", "10_model")
        if train == 1:
            data.to_csv("./GBM/need_label_" + site_name + ".csv")
            return 3
        if predict_encode == 0: return 2
        predict_encode = binary(str(predict_encode), 9)
    else: best = auto_brute(lock, jobs, done, encode_col, unique_mt, MINIMAL_REPEAT, model_predict)
    
    # Using prediction encoding to build suffix tree and segment data block

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

    whole_string, segments, record_seg, all_seqs = encode_and_segment(lock, encode_col, encode_option, unique_mt, ignore_len, MINIMAL_REPEAT)
    
    # MSA

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
        #for i in all_seqs[seg_idx][:-1]: print(i, "\n\t\t-> ", trans_dict[i])
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

        # Set txt & json
        with open(os.path.join(current_path, "Set-" + str(seg_idx+1) + ".txt"), 'w', encoding='utf-8') as file:
            for page in range(len(others)):
                #json_page = []
                output_dict = {} # Record which pattern is used
                if drop_last == 1:
                    length = len(all_seqs[seg_idx]) - 1
                else:
                    length = len(all_seqs[seg_idx])
                for s in range(length):
                    write_tmp = []
                    json_set = []
                    write_tmp.append(str(seg_idx+1) + "-" + str(page) + "-" + str(s+1) + "\t")
                    json_set.append(str(seg_idx+1) + "-" + str(page) + "-" + str(s+1))
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
                                    json_schema[c]["Encoding"] = encoding[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
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
                                    json_schema[c]["Encoding"] = encoding[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                    json_schema[c]["CECId"] = cecid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                    json_schema[c]["TECId"] = tecid[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                    json_schema[c]["ColType"] = col[tmp[output_dict[record_seg[seg_idx][1][1]]]+idx]
                                idx += 1
                        if len(list(c for c in write_tmp if c != '\t' and c != '')) != 1:
                            for word in write_tmp:
                                file.write(word)
                            file.write('\n')
                    #json_page.append(json_set)
                    json_result.append(json_set)
                #json_result.append(json_page)
        with open(os.path.join(current_path, "Set-" + str(seg_idx+1) + ".json"), 'w') as json_file:
            json.dump(json_result, json_file)
        with open(os.path.join(current_path, "SchemaSet-" + str(seg_idx+1) + ".json"), 'w') as json_file:
            json.dump(json_schema, json_file)
    
    # Modified TableA json Output

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
                    schema_dict["Encoding"] = -1
                    schema_dict["TECId"] = "1"
                    schema_dict["ColType"] = "Set-" + str(set_count)
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
    
    # Save good encoding for next use

    with open('./good_encode.txt', 'rb') as f:
        candidate = pickle.load(f)
    if encode_option not in candidate and brute == 1:
        with open('./good_encode.txt', 'wb') as f:
            candidate.append(encode_option)
            print("Append:", encode_option)
            pickle.dump(candidate, f)

    #print(candidate)

    if brute == 1:
        with open('./good_encode.txt', 'rb') as f:
            candidate = pickle.load(f)

    #candidate = []

    if brute == 1:
        with open('./good_encode.txt', 'wb') as f:
            pickle.dump(candidate, f)

    # Show model selected encoding

    if model_predict == 1:
        print(len(all_seqs), "Set\nModel Select: ", predict_encode, predict_ign_len)
    else:
        print(len(all_seqs), "Set\nBEST: ", best)
    print(record_seg)

    #cols = ["tag", "ids", "classes", "pathid", "parentid", "tecid", "cecid", "encoding", "col"]
    #for c in range(len(best['option'])):
    #    if best['option'][c] == '1':
    #        print(cols[c])
    return 0


# In[ ]:


if __name__ == "__main__":
    s = main()
    if s == 1: print("MC Occur, PASS")
    elif s == 2:
        os.system("mv ./GBM/test.csv ./GBM/need_label_" + site_name + ".csv")
        print("Model no suggest, rename test file to need_label.csv")
    elif s == 3: print("Train file created, name: need_label_" + site_name + ".csv")


# In[ ]:




