import glob
import json
import pdb

def merge():
    # topks = [20, 50, 100]

    topks = [100]
    
    for topk in topks:
        filepaths = glob.glob(f'data/ms2/results/raw/word_scores/{topk}/scores_dict_merge_*.json')
        # merge these files into one file by directly concat the lines
        
        # # generate str 1 to 1222
        # num_str = set([str(i) for i in range(0, 1223)])

        with open(f'data/ms2/results/scores_dict_merge_{topk}.json', 'w') as f:
            for filepath in filepaths:
                with open(filepath, 'r') as f1:
                    for line in f1:
                        data = json.loads(line)
                        # each time remove the key from num_str
                        # try:
                        #     num_str.remove(data['key'])
                        # except:
                        #     pdb.set_trace()
                        f.write(line)
            
        # # check if num_str is empty
        # if len(num_str) != 0:
        #     print(f'Error: {num_str}')


if __name__ == '__main__':
    merge()