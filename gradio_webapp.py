import gradio as gr
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

import numpy as np
import torch
from torchvision import datasets
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


import pandas as pd
import numpy as np 
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import pickle
from annoy import AnnoyIndex

import nltk

from tokenizer import LemmatizerTokenizer

# A fake dataframe with paths to images

image_list = np.sort(os.listdir(os.path.join("MLP-20M","MLP-20M")))
image_path_list = np.array([os.path.join("MLP-20M","MLP-20M",i) for i in image_list])

movie_dataset=pd.read_csv('movies_metadata.csv',dtype=str,encoding='utf-8', on_bad_lines='skip',lineterminator='\n')
df=movie_dataset[['id','title','release_date','overview']]


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')




class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _= super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path


class MobilenetFeatures(nn.Module):

  def __init__(self):
    super().__init__()
    self.features = mobilenet.features
    self.avg = nn.AdaptiveAvgPool2d(1)
    self.flatten_layer = nn.Flatten()

  def forward(self,x):
    x = self.features(x)
    x = self.avg(x)
    x = self.flatten_layer(x)
    return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_reco = torch.load('./mobilenet',map_location=torch.device('cpu'))

model_dist = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

path_vectorizer='tokenizer_instance.pkl'
with open(path_vectorizer,'rb') as file:
    vectorizer=pickle.load(file)


    #model = MobilenetFeatures()
model_reco.eval()



def process_image(image):

    

    image = Image.fromarray(image.astype('uint8'))

    
    

    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    normalize = transforms.Normalize(mean, std)
    inv_normalize = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std]    )

    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),normalize])

    #X  = ImageAndPathsDataset(image,transform)
    tensor = transform(image)

    features = model_reco(tensor.unsqueeze(0))

    vector = features.tolist()
    # Now we send the vector to the API
    # Replace 'annoy-db:5000' with your Flask server address if different (see docker-compose.yml)
    
    response = requests.post('http://127.0.0.1:5000/reco', json={'vector': vector})
    
    if response.status_code == 200:
        indices = response.json()


        # Retrieve paths for the indices
        listpaths = image_path_list[indices]

        print(listpaths)

        # Plot the images
        fig, axs = plt.subplots(1, len(listpaths), figsize=(5 * len(listpaths), 5))
        for i, path in enumerate(listpaths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis('off')
        return fig
    else:
        return "Error in API request"


def process_nlp_dist(user_text, method):

    



    if method == "Bag of Words":

        
        vector=vectorizer.transform([user_text])
        vector = vector.toarray().tolist()

        #print(np.shape(vector))


        response = requests.post('http://127.0.0.1:5000/text_bag', json={'vector': vector})

        indices = response.json()
        top_movies = df.iloc[indices]


        return pd.DataFrame(top_movies)





    elif method == "DistilBERT":

        if pd.isna(user_text):
            return np.zeros(model.config.dim)
        #model.to('cuda')
        inputs = tokenizer(user_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        #inputs = {key: value.to('cuda') for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model_dist(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        #print(vector)
        vector = vector.tolist()
        response = requests.post('http://127.0.0.1:5000/text_dist', json={'vector': vector})
        indices = response.json()
        top_movies = df.iloc[indices]
        return pd.DataFrame(top_movies)
    else:
        return "Please select a method"



# Function to process NLP with Gradio input
def process_nlp_with_gr_input(user_text, method):
    # Process the NLP input (replace this with your actual processing logic)
    processed_result = process_nlp_dist(user_text, method)

    # Display the processed result (replace this with your actual result display logic)
    return processed_result

# Main code to launch Gradio with your existing functions
iface_tab1 = gr.Interface(fn=process_image, inputs="image", outputs="plot")

iface_tab2 = gr.Interface(
    fn=process_nlp_with_gr_input,
    inputs=[gr.Textbox(lines=2, placeholder="Enter movie description here..."), 
            gr.Radio(["Bag of Words", "DistilBERT"], label="Select Method")],
    outputs=gr.Textbox()  # Use the default interpretation for the text box
)

iface = gr.TabbedInterface([iface_tab1, iface_tab2])
iface.launch()