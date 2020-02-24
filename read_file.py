def read_file(input_file_path):
    content = []
    recb_start = 0
    recb_end = 0
    tag = []
    ids = []
    classes = []
    pathid = []
    parentid = []
    tecid = []
    cecid = []
    encoding = []
    col = []
    others = []
    with open(input_file_path, encoding='utf-8') as file:
    #with open(input_file_path) as file:
        line = file.readline()
        while(line):
            line = file.readline()
            tmp = line.split('\t')
            if tmp[0] == "Content":
                content = tmp[1:-1]
            elif tmp[0] == "RecB":
                tmp = tmp[1:-1]
                for index in range(len(tmp)):
                    if tmp[index] == "SDR":
                        recb_start = index
                    elif tmp[index] == "EDR":
                        recb_end = index
            elif tmp[0] == "Tag":
                tag = tmp[1:-1]
            elif tmp[0] == "ID":
                ids = tmp[1:-1]
            elif tmp[0] == "Class":
                classes = tmp[1:-1]
            elif tmp[0] == "PathId":
                pathid = tmp[1:-1]
            elif tmp[0] == "ParentId":
                parentid = tmp[1:-1]
            elif tmp[0] == "TECId":
                tecid = tmp[1:-1]
            elif tmp[0] == "CECId":
                cecid = tmp[1:-1]
            elif tmp[0] == "Encoding":
                encoding = tmp[1:-1]
            elif tmp[0] == "Col Type":
                col = tmp[1:-1]
            else:
                others.append(tmp[1:-1])
    others = others[0:-4]
    return [content, recb_start, recb_end, tag, ids, classes, pathid, parentid, tecid, cecid, encoding, col, others]