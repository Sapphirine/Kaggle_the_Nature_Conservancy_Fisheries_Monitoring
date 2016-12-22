import pandas as pd
import datetime

main = pd.read_csv('submission_2016-12-13-01-58_LB_1.08602.csv')
LAG = pd.read_csv('submission_red_crop_fish_2016-12-13-10-26.csv')

target_main = main[main['LAG']>0.1]['image'].tolist()
target_LAG = LAG[LAG['LAG']>0.1]['image'].tolist()
target = set(target_main)&set(target_LAG)

target_index = []
for i in target:
    target_index.append(main[main['image']==i].index.values[0])

for i in target_index:
    main.iloc[target_index[0],:8]/=7
    main.iloc[i,3] = 1-sum(c.iloc[i,[0,1,2,4,5,6,7]])
    
now = datetime.datetime.now()
sub_file = 'weighted_submission' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
main.to_csv(sub_file, index=False)