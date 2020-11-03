from utils import *

if __name__ == '__main__':
    parallel = False
    # launch the pipeline
    # declare subjects data
    # subjects = ['ZF36621', 'ZF44516', 'ZF49582', 'ZF53093', 'ZF59543', 'ZF21147', 'ZF39497']
    subjects = ['stanford_hardi']

    # start fiber-tracking
    arr = subjects
    res = list()
    if parallel:
        p = Pool(processes=min(7, len(arr)))
        res = p.map(fiber_tracking, arr)
    else:
        for sub in subjects:
            res.append(fiber_tracking(sub))

    # declare dictionary for storing results
    res_table = dict()
    for table in res:
        subject = table["subject"]
        res_table[subject] = table
    
    # extract desired streamlines
    medium = [1, 2]
    area_pairs = [
        (11, 54),
    #     (54, 74),
    #     (28, 74),
        # (28, 54),
    #     (37, 80),
    #     (32, 76),
    ]
    for subject in res_table:
        save_results(res_table[subject], area_pairs, medium, subject)

    print('Saving Complete.')