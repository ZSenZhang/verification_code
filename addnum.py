
fh = open('train.txt','r')
f=open('new_train.txt','w')
i=-1
for line in fh:
    line=line.strip('\n')
    line = line.rstrip()
    words = line.split()
    # img = Image.open(words[0]) 
    # img2=Image.new('RGB',(maxheight,maxheight),(0,0,0))
    # if(img.height<maxheight):
    #     box1=(0,0,img.width,img.height)
    # else:
    #     box1=(0,0,img.width,maxheight)
    # region=img.crop(box1)
    # img2.paste(region,(0,0))
    # if(not os.path.exists(root+'new_'+path+'/')):
    #     os.makedirs(root+'new_'+path+'/')
    # img2.save(root+'new_'+path+'/'+str(i)+'.jpg')
    if(words[0]=='.'):
        continue
    if(words[0]=='..'):
        continue
    else:
        f.write('./class1+2+3_train/'+words[0]+' '+words[0][0]+'\n')
