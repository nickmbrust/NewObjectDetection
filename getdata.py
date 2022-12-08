import torch
import rave as r
import features as f
import featureExtractor as FE
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