from flask import Flask, request, jsonify
from annoy import AnnoyIndex


app = Flask(__name__)

# Load the Annoy database
annoy_db = AnnoyIndex(576, metric='angular')  # Here 2 is the dimension of the vectors in the database in my example
                                            # you would replace it with the dimension of your vectors
annoy_db.load('rec_imdb.ann')  # Replace 'annoy_db.ann' with the path to your Annoy database

annoy_db_nlp_d = AnnoyIndex(768, metric='angular')

annoy_db_nlp_d.load('distilbert_embeddings.ann')

annoy_db_nlp_bag = AnnoyIndex(500, metric='angular')

annoy_db_nlp_bag.load('tfidf_annoy_db.ann')



@app.route('/') # This is the home route, it just returns 'Hello world!'
def index():    # I use it to check that the server is running and accessible it's not necessary
    return 'Hello world!'

@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    vector = request.json['vector'] # Get the vector from the request
    closest_indices = annoy_db.get_nns_by_vector(vector[0], 5) 
    reco = [closest_indices[0], closest_indices[1],closest_indices[2],closest_indices[3],closest_indices[4]]  # Assuming the indices are integers
    return jsonify(reco) # Return the reco as a JSON


@app.route('/text_dist', methods=['POST'])
def text_dist():
    vector = request.json['vector'] # Get the vector from the request
    indices = annoy_db_nlp_d.get_nns_by_vector(vector, 5)
    
    return jsonify(indices)

@app.route('/text_bag', methods=['POST'])
def text_bag():
    vector = request.json['vector'] # Get the vector from the request
    #print(len(vector))
    
    indices = annoy_db_nlp_bag.get_nns_by_vector(vector[0], 5)
    
    return jsonify(indices)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Run the server on port 5000 and make it accessible externally
