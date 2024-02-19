import os
for i in range(10):
    #cmd = 'python mainlight.py --model efficientViT --bs 32 --lr 0.0002 --fold %d' %(i+1)
    #cmd = 'python mainlight.py --model MobileVit --bs 48 --lr 0.0001 --fold %d' %(i+1)
     cmd = 'python combinemodel.py --model Ourmodel --bs 128 --lr 0.0001 --fold %d' %(i+1)
     os.system(cmd)
print("Train efficientViT ok!")
os.system('pause')

