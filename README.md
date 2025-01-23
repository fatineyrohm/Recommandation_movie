This project is based on the project class framework and is designed to recommend movies using either images or text.

**Databases:**
Due to the size of the pickle and Annoy databases, they cannot be hosted online. However, we used the dataset available at 
**https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset**, which we trained using the following models:
   - **rec_imdb.ann**: image embedding generated Convolutional model 
   - **distilbert_embeddings.ann** : text embeddings generated  BERT model 
   - **tfidf_annoy_db.ann** : text embeddings generated TF-IDF text embeddings 

**Components:**

 - **gradio_web_app.py:**  A Gradio-based user interface that allows users to recommend 5 movies using either images or text. 
    Users can upload an image or enter a movie description, and results are shown either as images or as a table containing movie metadata.

- **annoy_api.py:** A Flask API that uses Annoy (Approximate Nearest Neighbor) to quickly search for similar movies across three pre-trained databases:

The API returns the indices of the top 5 most similar movies based on the input.
