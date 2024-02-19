import os
for i in range(10):
     cmd = 'python combinemodelkmu.py --model Ourmodel --bs 128 --lr 0.001 --fold %d' %(i+1)
     os.system(cmd)
print("Train ShuffViT ok!")
os.system('pause')

