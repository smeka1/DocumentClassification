
from flask import Flask,render_template, request, jsonify
import classify_csv as clfpy
from flask_cors import CORS
import json 

app = Flask(__name__)
CORS(app)

@app.route('/api')
@app.route('/api/')
def index():
   return """
   <html> 
   <h2>Welcome to Sriram's API!</h2>
   <body><h3>These are your available endpoints:</h3><br/>
   <a href='/api/classify'>Run requests to this URI</a> 
   <br/>   
   <a href='/api/classify_form'>Use input field here</a>
   </body>
   </html>
   """

@app.route('/api/classify',methods = ['POST', 'GET'])
def classify():
   if request.method == 'POST':
      content = request.form.get('words')
      if(request.headers.get('Content-Type') == 'application/json'):
         #json.dump(request.get_json())
         content = request.get_json().get('words')
         return  jsonify( prediction=clfpy.predict([content]))
      else:
         #print(content)
         if(content is None):
            return "Enter value for parameter:words."
         return jsonify( prediction=clfpy.predict([content])) 
   else:
      content = request.args.get('words')
      if(content is None):
         return "Please enter value for parameter:words. Use CURL or similar methods."
      if(request.headers.get('Content-Type') == 'text/html'):
         #print(content)
         return jsonify (prediction= clfpy.predict([content]))
      else:
         return jsonify (prediction=clfpy.predict([content]))
   

@app.route('/api/classify_form')
def classifyDoc():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0',port='3001',debug = True)
