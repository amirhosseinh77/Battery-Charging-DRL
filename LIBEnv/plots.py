import cv2
import numpy as np

def plot_SOC(SOC):
    MARGIN = 5
    img = np.ones((200,100,3), dtype='uint8')*255
    soc = np.array(range(MARGIN,int(SOC*100)))
    green = (soc*255/100).astype(int)
    red = 255-green
    y = (-200/100*soc + 200).astype(int) + MARGIN
    for i in range(1,len(soc)): img[y[i]:y[i-1],MARGIN:-MARGIN] = [0,green[i],red[i]] 
    img = cv2.rectangle(img, (5,10), (95,195), 0, MARGIN)
    img = cv2.rectangle(img, (40,0), (60,10), 0, -1)
    cv2.putText(img, f'{SOC:0.02f}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    return img

def plot_pack_SOC(SOCs):
    pack_imgs = []
    for soc in SOCs.ravel():
        pack_imgs.append(plot_SOC(soc))
    pack_imgs = np.hstack(pack_imgs)
    return pack_imgs
