import torch
import rave as r
import features as f
import featureExtractor as FE
from dataloader import loadimgs, loadClass
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")


def getSWSLrave():
    PATH = 'dataset'
    pathtest = PATH+ 'test/'

    positive = torch.load('SWSLclassfeats/n02772753.tar', map_location=device)
    postivefeats = positive['features']
    positivey = torch.ones((postivefeats.shape[0],1), dtype=torch.float)
    positivey = torch.tensor(positivey).to(device)
    postiverave = r.RAVE()


    with torch.no_grad():
        postiverave.add(postivefeats.to(device), positivey)
        print(postiverave.mxx.shape)

    negativerave = r.RAVE()   
    negativeclass = torch.load('SWSLclassfeats/n04045857.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n02853991.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n03266479.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n03276921.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n03443167.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n03802912.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n04076546.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n04190372.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('SWSLclassfeats/n04513584.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey


    averages = r.RAVE()
    averages.add_rave(postiverave)
    averages.add_rave(negativerave)
    return averages, postiverave, negativerave

def getCliprave():
    PATH = 'dataset'
    pathtest = PATH+ 'test/'
    swsltransformer = f.swsl_transform(128)

    positive = torch.load('Clipclassfeatures/n02772753.tar', map_location=device)
    postivefeats = positive['features']
    positivey = torch.ones((postivefeats.shape[0],1), dtype=torch.float)
    positivey = torch.tensor(positivey).to(device)
    postiverave = r.RAVE()


    with torch.no_grad():
        postiverave.add(postivefeats.to(device), positivey)
        print(postiverave.mxx.shape)

    negativerave = r.RAVE()   
    negativeclass = torch.load('Clipclassfeatures/n04045857.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n02853991.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n03266479.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n03276921.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n03443167.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n03802912.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n04076546.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))

    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n04190372.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey
    negativeclass = torch.load('Clipclassfeatures/n04513584.tar', map_location=device)
    negativefeats = negativeclass['features']
    negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
    negativerave.add(negativefeats.to(device), -negativeey.to(device))
    del negativeclass, negativefeats, negativeey


    averages = r.RAVE()
    averages.add_rave(postiverave)
    averages.add_rave(negativerave)

    return averages, postiverave, negativerave
def gettrainingdata(feattype):
    PATH = 'dataset/'
    pathtrain = PATH+ 'training/'
    ytrain = []
    if feattype == 'swsl':
        transform = f.swsl_transform(128)
        imgs, labelstrain = loadimgs(pathtrain, transform, 'swsl')
        Xtrain = FE.swslextract(imgs)
    if feattype == 'clip':
        transform = f.Clip_transform(228)
        imgs, labelstrain = loadimgs(pathtrain, transform, 'clip')
        Xtrain = FE.newclip(imgs)
    for h in range(len(labelstrain)):
        if labelstrain[h] == "n02772753":
            ytrain.append(1.0)
        else:
            ytrain.append(-1.0)
    ytrain = torch.tensor(ytrain)
    return Xtrain, ytrain
    