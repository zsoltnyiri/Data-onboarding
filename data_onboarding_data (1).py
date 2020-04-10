from __future__ import print_function
import time 
import requests
import cv2
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import os

# import sys
# import pyscreenshot as ImageGrab
# from time import sleep

### Get data

_url = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize'
_key = 'e84596b3f9b64377904e2e896dde8f42'
_maxNumRetries = 10

def processRequest( json, data, headers, params ):

    retries = 0
    result = None

    while True:

        response = requests.request( 'post', _url, json = json, data = data, headers = headers, params = params )

        if response.status_code == 429: 

            print( "Message: %s" % ( response.json()['error']['message'] ) )

            if retries <= _maxNumRetries: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )

        break
        
    return result
    
def renderResultOnImage( result, img ):
    
    """Display the obtained results onto the input image"""
    
    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        cv2.rectangle( img,(faceRectangle['left'],faceRectangle['top']),
                           (faceRectangle['left']+faceRectangle['width'], faceRectangle['top'] + faceRectangle['height']),
                       color = (255,0,0), thickness = 3 )


    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        currEmotion = max(currFace['scores'].items(), key=operator.itemgetter(1))[0]


        textToWrite = "%s" % ( currEmotion )
        cv2.putText( img, textToWrite, (faceRectangle['left'],faceRectangle['top']-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3 )
        
# Load raw image file into memory

f = 0
for file in os.listdir("C:\Users\zsolt.nyiri\Desktop\pic"):
    if file.endswith(".jpg"):
        os.path.join("C:\Users\zsolt.nyiri\Desktop\pic", file)
        f +=1
        
data = {}
for i in range(f):
    pathToFileInDisk = r'C:\Users\zsolt.nyiri\Desktop\pic\host_pic_' + str(i) + '.jpg'
    with open( pathToFileInDisk, 'rb' ) as f:
        data[i] = f.read()

headers = dict()
headers['Ocp-Apim-Subscription-Key'] = _key
headers['Content-Type'] = 'application/octet-stream'

json = None
params = None

# Running the Emotion API on the pictures
result = {}
for i in range(len(data)):
    result[i] = processRequest(json, data[i], headers, params)

# Remove pictures where the no. of people does not equal the mode
person = {}    
for i in range(len(result)):
    person[i] = len(result[i])
    
person_mode = pd.Series(person.values()).mode()

for i in range(len(result)):
    if len(result[i]) != person_mode[0]:
        del result[i]
 
# Reset the dictionary indexes
result = {i: v for i, v in enumerate(result.values())}

if max(range(len(result))) > 0:
    results = [pd.DataFrame() for i in range(max([len(pic) for k,pic in result.iteritems()]))]
    for pic_count in range(len(result)):
        for person in range(len(result[pic_count])):
            results[person] = results[person].append(result[pic_count][person]['scores'],
                           ignore_index = True)
        
else:
    results = pd.DataFrame()
    for row in range(len(result[0])):
        results = results.append(result[0][row]['scores'], ignore_index = True)
    
# Max happiness
max_hep = {}
for i in range(len(results)):    
    max_hep[i] = max(results[i]['happiness'])

max_hep_loc = {}
for i in range(len(max_hep)):    
    max_hep_loc[i] = results[i][results[i]['happiness'] == max_hep[i]].index.tolist()   
 

# Show the most happy images for every person
if result is not None:
    # Load the original image from disk
    for i in range(len(max_hep_loc)):
        data8uint = np.fromstring( data[max_hep_loc[i][0]], np.uint8 ) # Convert string to an unsigned int array
        img = cv2.cvtColor(cv2.imdecode( data8uint, cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB)
    
        renderResultOnImage(result[max_hep_loc[i][0]], img)
    
        ig, ax = plt.subplots(figsize=(15, 20))
                
        ax.set_title('Person_' + str(i+1) + "'s happiest picture", fontdict = {'fontsize':15})      
        ax.imshow(img)
 
## Reconstructing the data for better visualization
# Calculating per person averages
means = pd.DataFrame()
for i in range(len(results)):
    means = means.append(np.mean(results[i]), ignore_index = True)

''''
## Calculate whole group average instead of avg. per person    
# Might be useful when there is too many people, currently unused
mm = pd.DataFrame()
mm = np.mean(means)
mm = pd.DataFrame(mm)
mm['index'] = (0)
mm['emotion'] = mm.index
mm = mm.pivot(index = 'index',  columns = 'emotion', values = 0)
'''

# Viz

# Create traces for each person in the pic
if max(range(len(result))) > 0:
    my_dict = {}
    for x in range(len(results)): 
        my_dict[x] = go.Bar(
            x = list(means.columns.values),
            y = list(means.iloc[x].values*100),
            name = "Person_" + str(x+1)
            )   

# Use grouped barchart             
    layout = go.Layout(
        title = 'Per person averages',
        xaxis = dict(tickangle = -45),
        yaxis = dict(title = '%',
                     range = [0, 100]),
        barmode ='group'
        )

# Create the chart            
    py_data = [my_dict[i] for i in my_dict]
    fig = go.Figure(data = py_data, layout = layout)
    py.plot(fig, filename = 'group_average_bar.html')

# Different logic when there is only 1 person in the pic, since the result json is less nested
else:

    my_dict = {}   
    for x in range(len(results)):
        my_dict[x] = go.Bar(
        x = list(results.columns.values),
        y = list(results.iloc[x].values*100),
        name = "Person_" + str(x+1)
        )       
     
    layout = go.Layout(
        xaxis = dict(tickangle=-45),
        barmode ='group'
    )
    
    py_data = [my_dict[i] for i in my_dict]
    fig = go.Figure(data = py_data, layout = layout)
    py.plot(fig, filename = 'solo_bar.html')