import numpy as np

def name_to_minite(txt):
    hrs = int(txt[txt.find("H")-2:txt.find("H")])
    mint = int (txt[txt.find("Mp4")+4:txt.find("Min")-1])
    minute = (hrs*60+mint)
    return minute

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
def determine_threshod(frame,selection_scheme ='Best'):
    kmeans_data = {'middle':90.9737129,
              'day':0.501917,
              'night':0.260961 ,
              'day_mu_sigma':[0.501916839566159, 0.08075117723283297],
              'night_mu_sigma':[0.26096127350033066, 0.0848839574869915]}
    conf_thres=0
    if selection_scheme=='Best':
        print("Sorry, please provide file names")
    
    if selection_scheme =='Two_cluster':
        img10=frame
        B = img10[:, :, 0]
        G = img10[:, :, 1]
        R = img10[:, :, 2]
        luminance =np.mean( 0.299 * R + 0.587 * G + 0.114 * B)
        if luminance > kmeans_data['middle']:
            conf_thres = kmeans_data['day']
        if luminance < kmeans_data['middle']:
            conf_thres = kmeans_data['night']
    if selection_scheme=='Dynamic':
        img10=frame
        B = img10[:, :, 0]
        G = img10[:, :, 1]
        R = img10[:, :, 2]
        luminance =np.mean( 0.299 * R + 0.587 * G + 0.114 * B)
        if luminance > kmeans_data['middle']:
            conf_thres = np.random.normal(kmeans_data['day_mu_sigma'][0], kmeans_data['day_mu_sigma'][1], 1) 
        if luminance < kmeans_data['middle']:
            conf_thres = np.random.normal(kmeans_data['night_mu_sigma'][0], kmeans_data['night_mu_sigma'][1], 1) 
        conf_thres=conf_thres
        if conf_thres<=0: conf_thres = 0.01
    return conf_thres
# print (line_intersection((A, B), (C, D)))