import os
for i in range(10):
    
     cmd = 'python combinemodelkdef.py --model Ourmodel --bs 16 --lr 0.0001 --fold %d' %(i+1)
     os.system(cmd)
print("Train ourmodel ok!")
os.system('pause')
