from flask import Flask, request, jsonify, abort
from src.predict_scripts.emulated_pipeline import predict_class
app = Flask(__name__)             
@app.route("/")                  
def hello():                     
    return jsonify(message="Hello from prediction");    

@app.route("/predict", methods=['POST'])                  
def predict():                      
    body = request.json  
    if("text" in body):
        accuracy = predict_class(str(body["text"]))
        return jsonify(error=False,accuracy=2), 200
    else:
        return jsonify(error=True,description="Property text not defined"), 400
        
if __name__ == "__main__":        
    app.run(port=3001)                     