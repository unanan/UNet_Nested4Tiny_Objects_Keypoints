import os
import glob
import cv2
try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

meterIDlist = ['n02794156','n03841143','n03891332']

def checkDataNum(anno_rootpath = 'D:\\ImageNet\\Meter3800-3\\Annotation', image_rootpath = 'D:\\ImageNet\\Meter3800-3\\Image'):
    datanumcount = 0
    for folder in os.listdir(image_rootpath):
        for file in os.listdir(os.path.join(image_rootpath,folder)):
            if not os.path.isdir(file):
                if os.path.isfile(os.path.join(anno_rootpath,folder,os.path.splitext(file)[0]+".xml")):
                    datanumcount+=1

    print(datanumcount)


def getCharacters(xmlpath):
    boxes = []
    # ismeter = []
    try:
        tree = ET.parse(xmlpath)     #??xml??
        root = tree.getroot()        #??root??

        width =0
        height =0
        for size in root.findall('size'):
            width = int(size[0].text)
            height = int(size[1].text)
            break

        for object in root.findall('object'):  # ??root??????country??
            if object[0].text in meterIDlist:
                xmin = object[4][0].text
                ymin = object[4][1].text
                xmax = object[4][2].text
                ymax = object[4][3].text
                boxes.append([1,int(xmin),int(ymin),int(xmax),int(ymax)])
            else:
                boxes.append([0,0,0,0,0])
    except Exception as e:
        print(xmlpath)
        print(e)

        #print "Error:cannot parse file:country.xml."

    return boxes


def showBbox(anno_rootpath = 'D:\\ImageNet\\Meter3800-3\\Annotation', image_rootpath = 'D:\\ImageNet\\Meter3800-3\\Image'):
    for folder in os.listdir(anno_rootpath):
        for file in os.listdir(os.path.join(anno_rootpath,folder)):
            if os.path.isfile(os.path.join(image_rootpath,folder,os.path.splitext(file)[0]+".JPEG")):
                boxes = getCharacters(os.path.join(anno_rootpath,folder,file))
                img = cv2.imread(os.path.join(image_rootpath,folder,os.path.splitext(file)[0]+".JPEG"))
                for box in boxes:
                    if box[0]:
                        cv2.rectangle(img, (box[1], box[2]), (box[3], box[4]), (0, 255, 0), 2)
                cv2.imshow("meter", img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

# checkDataNum()
showBbox()
