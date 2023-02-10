import re
import json
def get_the_label_and_version(file_path):
    version_map ={}
    '''
    {
        'label':[upgrade_package_label]
    }
    '''
    lib_name_version = {}
    '''
    {
        lib_name:{
            "version":labelId
        }
    }
    '''
    name_version_pair = []
    with open(file_path,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('__label__'):
            line = line[9:]
            idx=0
            i = 0
            for l in line:
                if l.isdigit():
                    idx *=10
                    idx += int(l)
                    i +=1
                else:
                    break
            # if idx == 1724:
            #     print(line)
            line = line[i+2:]
            lib_name_without_version = re.sub(r'[0-9]+','',line)
            # lib_version = re.findall(r"\d+",line)
            lib_version = find_version(line)

            # if len(lib_version) == 0:
            #     lib_version = 0
            # else:
            #     lib_version = int(lib_version[0])

            

            name_version_pair.append((lib_name_without_version,lib_version,idx))
            

            if lib_name_without_version in lib_name_version:
                if lib_version in lib_name_version[lib_name_without_version]:
                    print(lib_version,lib_name_without_version)
                    print(line)
                    print('\n')
                lib_name_version[lib_name_without_version][lib_version] = idx
            else:
                lib_name_version[lib_name_without_version] = {
                    lib_version:idx
                }
        else:
            continue
    for l_n_w_v,l_v,idx in name_version_pair:
        '''construct the map'''
        for k,v in lib_name_version[l_n_w_v].items():
            if k < l_v:
                if v not in version_map:
                    version_map[v] = []
                version_map[v].append(idx)
    return version_map 


        
def find_version(name):
    '''x_x_x'''
    '''x_x'''
    '''
    x(first_ones or last ones)
    '''
    version = re.findall(r"\d+\_\d+\_\d+",name)
    if len(version) >= 1:
        version = version[0].split('_')
        num = 0
        for each in version:
            num *= 100
            num += int(each)
        return num
    if len(version) > 1:
        print(name)
    version = re.findall(r"\d+\_\d+",name)
    if len(version) >= 1:
        version = version[0].split('_')
        num = 0
        for each in version:
            num *= 100
            num += int(each)
        return num
    if len(version) > 1:
        print(name)
    version = re.findall(r"\d+",name)
    if len(version) == 0:
        num = 0
    else:
        num = int(version[0])
    return num




    



if __name__ == "__main__":
    version_map=get_the_label_and_version('../zero_shot_dataset/zestxml/Yf.txt')
    with open('version_map.json','w') as f:
        json.dump(version_map,f,indent=2)
    